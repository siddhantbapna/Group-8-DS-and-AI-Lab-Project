import argparse
import os

from config.config import paths
from src.train import train
from src.inference import run_inference


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", choices=["train", "infer"], default="train")
	parser.add_argument("--model", choices=["unet2d", "unet3d", "attenunet", "nnunet", "resunet", "vnet"], default="unet3d")
	parser.add_argument("--fold", type=int, default=0)
	parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
	parser.add_argument("--no-resume", action="store_true", help="Do not resume from latest checkpoint")
	parser.add_argument("--epochs", type=int, help="Override max epochs for this run")
	parser.add_argument("--input", type=str, help="Case directory for inference")
	parser.add_argument("--output", type=str, default=os.path.join(paths.predictions, "prediction.nii.gz"))
	args = parser.parse_args()

	if args.mode == "train":
		spatial_dims = 2 if args.model == "unet2d" else 3
		# Pass epochs override if provided
		train(model_name=args.model, spatial_dims=spatial_dims, resume_from=args.resume, max_epochs=args.epochs, no_resume=args.no_resume)
	else:
		if not args.input:
			raise SystemExit("--input is required in infer mode")
		ckpt = os.path.join(paths.models, f"best_{args.model}.pth")
		run_inference(args.input, ckpt_path=ckpt, model_name=args.model, output_path=args.output)
		print(f"Saved: {args.output}")


if __name__ == "__main__":
	main()
