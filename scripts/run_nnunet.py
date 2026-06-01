import argparse
import subprocess


def run_command(command):
    """打印后执行 nnU-Net 命令，日志里留痕。"""
    print("执行命令:", " ".join(command))
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description="nnU-Net v2 训练和预测命令封装。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="运行 nnUNetv2_plan_and_preprocess。")
    plan_parser.add_argument("--dataset-id", required=True)
    plan_parser.add_argument("--verify-dataset-integrity", action="store_true")

    train_parser = subparsers.add_parser("train", help="运行 nnUNetv2_train。")
    train_parser.add_argument("--dataset-id", required=True)
    train_parser.add_argument("--configuration", default="3d_fullres")
    train_parser.add_argument("--fold", required=True)
    train_parser.add_argument("--trainer", default="nnUNetTrainer")
    train_parser.add_argument("--plans", default="nnUNetPlans")

    predict_parser = subparsers.add_parser("predict", help="运行 nnUNetv2_predict。")
    predict_parser.add_argument("--input", required=True)
    predict_parser.add_argument("--output", required=True)
    predict_parser.add_argument("--dataset-id", required=True)
    predict_parser.add_argument("--configuration", default="3d_fullres")
    predict_parser.add_argument("--fold", required=True)
    predict_parser.add_argument("--trainer", default="nnUNetTrainer")
    predict_parser.add_argument("--plans", default="nnUNetPlans")

    args = parser.parse_args()

    if args.command == "plan":
        command = ["nnUNetv2_plan_and_preprocess", "-d", str(args.dataset_id)]
        if args.verify_dataset_integrity:
            command.append("--verify_dataset_integrity")
        run_command(command)

    elif args.command == "train":
        command = [
            "nnUNetv2_train",
            str(args.dataset_id),
            args.configuration,
            str(args.fold),
            "-tr",
            args.trainer,
            "-p",
            args.plans,
        ]
        run_command(command)

    elif args.command == "predict":
        command = [
            "nnUNetv2_predict",
            "-i",
            args.input,
            "-o",
            args.output,
            "-d",
            str(args.dataset_id),
            "-c",
            args.configuration,
            "-f",
            str(args.fold),
            "-tr",
            args.trainer,
            "-p",
            args.plans,
        ]
        run_command(command)


if __name__ == "__main__":
    main()
