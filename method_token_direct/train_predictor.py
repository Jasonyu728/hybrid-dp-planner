import os
import sys
import argparse
import torch
from torch import optim
from timm.utils import ModelEma
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


METHOD_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(METHOD_DIR)

# When this script is launched as `python method_token_direct/train_predictor.py`,
# Python puts `method_token_direct` before the project root in sys.path. That would
# shadow the real top-level `diffusion_planner` package with
# `method_token_direct/diffusion_planner`. Force the project root to be first.
sys.path = [
    p for p in sys.path
    if os.path.abspath(p or os.getcwd()) != METHOD_DIR
]
if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.train_utils import set_seed, save_model, resume_model
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.utils.lr_schedule import CosineAnnealingWarmUpRestarts
from diffusion_planner.utils.tb_log import TensorBoardLogger as Logger
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.dataset import DiffusionPlannerData
from diffusion_planner.utils import ddp

from method_token_direct.diffusion_planner.train_epoch import train_epoch


def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    parser = argparse.ArgumentParser(description="Direct token-classification training")
    parser.add_argument("--name", type=str, default="method-token-direct")
    parser.add_argument("--save_dir", type=str, default=".")

    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument("--nbr_vocab_path", type=str, default=None)

    parser.add_argument("--train_set", type=str, default=None)
    parser.add_argument("--train_set_list", type=str, default=None)

    parser.add_argument("--future_len", type=int, default=80)
    parser.add_argument("--time_len", type=int, default=21)
    parser.add_argument("--agent_state_dim", type=int, default=11)
    parser.add_argument("--agent_num", type=int, default=32)
    parser.add_argument("--static_objects_state_dim", type=int, default=10)
    parser.add_argument("--static_objects_num", type=int, default=5)
    parser.add_argument("--lane_len", type=int, default=20)
    parser.add_argument("--lane_state_dim", type=int, default=12)
    parser.add_argument("--lane_num", type=int, default=70)
    parser.add_argument("--route_len", type=int, default=20)
    parser.add_argument("--route_state_dim", type=int, default=12)
    parser.add_argument("--route_num", type=int, default=25)

    parser.add_argument("--augment_prob", type=float, default=0.5)
    parser.add_argument("--normalization_file_path", type=str, default="normalization.json")
    parser.add_argument("--use_data_augment", default=True, type=boolean)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--pin-mem", action="store_true")
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train_epochs", type=int, default=500)
    parser.add_argument("--save_utd", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--warm_up_epoch", type=int, default=5)
    parser.add_argument("--encoder_drop_path_rate", type=float, default=0.1)
    parser.add_argument("--decoder_drop_path_rate", type=float, default=0.1)

    # 本方法只保留 diffusion loss 和 token 编号分类 loss。
    parser.add_argument("--alpha_planning_loss", type=float, default=1.0)
    parser.add_argument("--lambda_token_cls_ce", type=float, default=1.0)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_ema", default=True, type=boolean)

    parser.add_argument("--encoder_depth", type=int, default=3)
    parser.add_argument("--decoder_depth", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--diffusion_model_type", type=str, choices=["score", "x_start"], default="x_start")
    parser.add_argument("--token_emb_dim", type=int, default=64)
    parser.add_argument("--learnable_token_emb", default=False, type=boolean)
    parser.add_argument("--use_token_classifier", default=True, type=boolean)
    parser.add_argument("--token_selection_mode", type=str, choices=["nearest", "classifier"], default="classifier")

    parser.add_argument("--predicted_neighbor_num", type=int, default=10)
    parser.add_argument("--resume_model_path", type=str, default=None)

    parser.add_argument("--use_wandb", default=False, type=boolean)
    parser.add_argument("--notes", default="", type=str)

    parser.add_argument("--ddp", default=True, type=boolean)
    parser.add_argument("--port", default="22323", type=str)

    args = parser.parse_args()

    args.state_normalizer = StateNormalizer.from_json(args)
    args.observation_normalizer = ObservationNormalizer.from_json(args)
    args.guidance_fn = None

    if args.learnable_token_emb:
        raise ValueError("method_token_direct expects --learnable_token_emb False.")
    if not args.use_token_classifier or args.token_selection_mode != "classifier":
        raise ValueError(
            "method_token_direct expects --use_token_classifier True "
            "and --token_selection_mode classifier."
        )

    return args


def model_training(args):
    global_rank, rank, _ = ddp.ddp_setup_universal(True, args)

    if global_rank == 0:
        print(f"------------- {args.name} -------------")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Use device: {args.device}")
        print("Method: direct token classification")

        if args.resume_model_path is not None:
            save_path = args.resume_model_path
        else:
            from datetime import datetime, timezone, timedelta

            time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d-%H:%M:%S")
            save_path = f"{args.save_dir}/training_log/{args.name}/{time}/"
            os.makedirs(save_path, exist_ok=True)

        args_dict = vars(args)
        args_dict = {
            k: v if not isinstance(v, (StateNormalizer, ObservationNormalizer)) else v.to_dict()
            for k, v in args_dict.items()
        }

        from mmengine.fileio import dump

        dump(args_dict, os.path.join(save_path, "args.json"), file_format="json", indent=4)
    else:
        save_path = None

    set_seed(args.seed + global_rank)

    aug = StatePerturbation(augment_prob=args.augment_prob, device=args.device) if args.use_data_augment else None
    train_set = DiffusionPlannerData(
        args.train_set,
        args.train_set_list,
        args.agent_num,
        args.predicted_neighbor_num,
        args.future_len,
    )
    train_sampler = DistributedSampler(
        train_set,
        num_replicas=ddp.get_world_size(),
        rank=global_rank,
        shuffle=True,
    )

    loader_kwargs = dict(
        dataset=train_set,
        sampler=train_sampler,
        batch_size=args.batch_size // ddp.get_world_size(),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(**loader_kwargs)

    if global_rank == 0:
        print(f"Dataset Prepared: {len(train_set)} train data\n")

    if args.ddp and torch.distributed.is_initialized():
        torch.distributed.barrier()

    diffusion_planner = Diffusion_Planner(args)
    diffusion_planner = diffusion_planner.to(rank if args.device == "cuda" else args.device)

    if args.ddp:
        diffusion_planner = DDP(diffusion_planner, device_ids=[rank], find_unused_parameters=False)
        diffusion_planner._set_static_graph()

    model_ema = None
    raw_model = ddp.get_model(diffusion_planner, args.ddp)
    if args.use_ema:
        model_ema = ModelEma(raw_model, decay=0.999, device=args.device)

    if global_rank == 0:
        print(f"Model Params: {sum(p.numel() for p in raw_model.parameters())}")

    optimizer = optim.AdamW(
        [{"params": raw_model.parameters(), "lr": args.learning_rate}]
    )
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, args.train_epochs, args.warm_up_epoch)

    if args.resume_model_path is not None:
        print(f"Model loaded from {args.resume_model_path}")
        diffusion_planner, optimizer, scheduler, init_epoch, wandb_id, model_ema = resume_model(
            args.resume_model_path,
            diffusion_planner,
            optimizer,
            scheduler,
            model_ema,
            args.device,
        )
    else:
        init_epoch = 0
        wandb_id = None

    wandb_logger = Logger(args.name, args.notes, args, wandb_resume_id=wandb_id, save_path=save_path, rank=global_rank)

    if args.ddp and torch.distributed.is_initialized():
        torch.distributed.barrier()

    for epoch in range(init_epoch, args.train_epochs):
        train_sampler.set_epoch(epoch)
        if global_rank == 0:
            print(f"Epoch {epoch + 1}/{args.train_epochs}")

        train_loss, train_total_loss = train_epoch(train_loader, diffusion_planner, optimizer, args, model_ema, aug)

        if global_rank == 0:
            wandb_logger.log_metrics({f"train_loss/{k}": v for k, v in train_loss.items()}, step=epoch + 1)
            wandb_logger.log_metrics({"lr/lr": optimizer.param_groups[0]["lr"]}, step=epoch + 1)

            if (epoch + 1) % args.save_utd == 0:
                ema_for_save = model_ema.ema if model_ema is not None else ddp.get_model(diffusion_planner, args.ddp)
                save_model(
                    diffusion_planner,
                    optimizer,
                    scheduler,
                    save_path,
                    epoch,
                    train_total_loss,
                    wandb_logger.id,
                    ema_for_save,
                )
                print(f"Model saved in {save_path}\n")

        scheduler.step()


if __name__ == "__main__":
    model_training(get_args())
