import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from crossec_forecast import (
    DataConfig,
    TrainConfig,
    BenchmarkConfig,
    build_dataloaders,
    BenchmarkEngine,
    seed_everything,
)
from examples.mock_panel_data import generate_mock_panel_data


def main():
    seed_everything(42)

    # 1. Prepare Mock Data (mimicking standar_panel.csv)
    mock_csv_path = Path("./dataset/mock_standar_panel.csv")
    mock_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("\n[Step 1/3] Generating realistic financial panel data...")
    df = generate_mock_panel_data(
        output_path=str(mock_csv_path),
        num_timestamps=150,
        seed=42,
    )

    # 2. Build Zero-Leakage DataLoaders with L=6 lookback window & Embargo
    print("\n[Step 2/3] Building DataLoaders with lookback L=6 and Embargo isolation...")
    data_config = DataConfig(
        target_col="logret1_win",
        fwd_ret_col="fwd_logret_1",
        timestamp_col="timestamp",
        symbol_col="symbol",
        seq_len=6,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        embargo_steps=1,
        batch_size=64,
        shuffle_train=True,
    )

    train_loader, val_loader, test_loader, meta_info = build_dataloaders(
        data=df,
        config=data_config,
    )

    print(f"DataLoaders successfully built!")
    print(f"- Feature Columns Count: {meta_info['num_features']}")
    print(f"- Lookback Sequence Length: {meta_info['seq_len']}")
    print(f"- Train Samples: {meta_info['n_train_samples']}")
    print(f"- Val Samples:   {meta_info['n_val_samples']}")
    print(f"- Test Samples:  {meta_info['n_test_samples']}")

    # 3. Configure Training Engine (Stage 1 Backprop & Stage 2 Rank IC early stopping)
    train_config = TrainConfig(
        epochs=15,
        lr=1e-3,
        weight_decay=1e-4,
        grad_clip_norm=1.0,
        early_stopping_patience=5,
        loss_type="bce",
        device="auto",
        checkpoint_dir="./checkpoints",
    )

    # 4. Configure Multi-Model Benchmark (Plugin/Plug-out)
    benchmark_config = BenchmarkConfig(
        models=[
            {
                "name": "mlp",
                "config": {"hidden_dims": [64, 32], "dropout": 0.2, "use_norm": True},
            },
            {
                "name": "lstm",
                "config": {"hidden_dim": 48, "num_layers": 2, "dropout": 0.2, "pooling": "last"},
            },
            {
                "name": "dlinear",
                "config": {"kernel_size": 3, "individual": False},
            },
            {
                "name": "tsfm_wrapper",
                "config": {"d_model": 48, "nhead": 4, "num_layers": 2, "dropout": 0.1},
            },
        ],
        top_quantile=0.20,
        export_dir="./benchmark_reports",
    )

    # 5. Run Benchmark
    print("\n[Step 3/3] Executing Multi-Model Benchmark Evaluation...")
    engine = BenchmarkEngine(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        meta_info=meta_info,
        train_config=train_config,
        benchmark_config=benchmark_config,
    )

    summary_df = engine.run()
    print("\n>>> Final Multi-Model Benchmark Summary Table <<<")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

