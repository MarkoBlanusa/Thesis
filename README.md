<!-----––––––––––––––––––––––––––––––––––––––––––––––––––––––––––>
## 🎯 Goal

End-to-end research code for my Master’s thesis on **cryptocurrency
portfolio optimisation**.  
The repo covers **data ingestion → dataset engineering → benchmark
models → RL (PPO) trading agent → result notebooks** so that anyone can
re-run experiments or plug in new assets with minimal effort.

Thesis/
│
├── Project_code/                 ← all Python / notebook code
│   ├── data/                     ← ready-made datasets (CSV + HDF5)
│   ├── data_retrievers/          ← scripts/notebooks that build `data/`
│   │   ├── binance.py
│   │   ├── data_collector.py
│   │   ├── database.py
│   │   ├── main_data.py
│   │   ├── utils.py
│   │   ├── lunarcrush_retriever.ipynb
│   │   └── data_retriever.ipynb
│   ├── PPO_dataset_maker.py      ← turn raw OHLCV into PPO sequences
│   ├── garch_adcc_model.ipynb
│   ├── lstm_models.ipynb
│   ├── xgb_adcc_model.ipynb
│   ├── main_ray_portfolio.py
│   ├── trade_env_ray_portfolio.py
│   ├── main_test_ray_portfolio.py
│   └── test_trade_env_ray_portfolio.py
│
├── Results_final/                ← published metrics, plots, weights
├── Studies/                      ← academic papers that motivated work
├── requirements.txt / environment.yml
└── README.md                     ← **you are here**


🚀 Quick start
git clone https://github.com/MarkoBlanusa/Thesis.git
cd Thesis
git lfs pull                       # download ~4.7 GB of datasets

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt


🛠️ Data layer
Script / Notebook	Purpose	Output
main_data.py	Streams OHLCV candles from Binance and stores them in HDF5	data/binance.h5
PPO_dataset_maker.py	Windows & normalises data for PPO	data/*.npy
lunarcrush_retriever.ipynb	Social-sentiment via LunarCrush API	data/lunarcrush_features.csv
data_retriever.ipynb	Macro data from Yahoo + crypto subset	data/macro_crypto_features.csv

Tip: retrievers take CLI flags or notebook widgets so you can add symbols or adjust dates in seconds.

.

📊 Benchmark models
garch_adcc_model.ipynb – GJR-GARCH-ADCC baseline

lstm_models.ipynb – two multivariate LSTM variants

xgb_adcc_model.ipynb – gradient-boosted covariance model

Each notebook ingests the same CSV/HDF5 datasets and logs metrics to Results_final/.

🤖 Reinforcement Learning (PPO)
File	Role
main_ray_portfolio.py	Launches RLlib PPO training with YAML config
trade_env_ray_portfolio.py	Custom Gym env: multi-asset weights, fees
*_test_*.py	Verbose test harness with extra prints

Where do the RLlib/Tune results go?
By default, Ray writes every experiment to ~/ray_results/<trainable>/<timestamp>/… on the user’s home drive 
docs.ray.io
stackoverflow.com
.
You’ll find checkpoints (checkpoint_*), progress CSVs, and TensorBoard event files there. Start TensorBoard with:

bash
Copier
Modifier
tensorboard --logdir ~/ray_results
You can change the location by:

Passing local_dir="path/to/runs" to tune.Tuner / tune.run 
stackoverflow.com
; or

Setting storage_path via ray.air.RunConfig (Ray ≥2.3) 
discuss.ray.io
github.com
.

If you forget to set these, Ray may duplicate large folders in both the custom path and ~/ray_results (known issue)

📈 Results & reproducibility
All final plots, tables, and trained checkpoints that appear in the thesis live under Results_final/.
Delete the folder and rerun the pipelines to reproduce from scratch.

📝 Extending the project
New assets – add symbol to data_retriever.ipynb; regenerate CSV.

Different API – subclass a new client in data_retrievers/; call from main_data.py.

New RL algorithm – swap RLlib trainer in main_ray_portfolio.py; env stays unchanged.

⚙️ Troubleshooting
Issue	Fix
Push rejected > 2 GiB	Ensure file type is in .gitattributes so Git LFS handles it.
Ray fills home drive	Pass local_dir/storage_path or clean ~/ray_results periodically.
CUDA OOM	Lower train_batch_size in PPO YAML or run CPU-only.

📚 References
Annotated PDFs of the key academic papers are stored in Studies/.
