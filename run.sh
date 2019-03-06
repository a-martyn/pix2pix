python main.py --title baseline --norm_type instance --d_loss SCE
python main.py --title batch_norm --norm_type batch --d_loss SCE
python main.py --title mse --norm_type instance --d_loss MSE
python main.py --title mse_batch_norm --norm_type batch --d_loss MSE