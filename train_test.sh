# accelerate launch train_hf_trainer.py --folds 0,1,2,3,4 --epochs 100 --batch-size 2 --bf16 --output-mode exp1
# accelerate launch train_hf_trainer.py --folds 0,1,2,3,4 --epochs 100 --batch-size 2 --bf16 --output-mode exp4
accelerate launch train_hf_trainer_aux_test.py --folds 0,1,2,3,4 --epochs 50 --batch-size 2 --bf16 --output-mode exp4 --log1p-targets
# accelerate launch train_hf_trainer.py --folds 0,1,2,3,4 --epochs 50 --batch-size 2 --bf16 --output-mode exp6 --no-nonneg-clamp
# accelerate launch train_hf_trainer.py --folds 0,1,2,3,4 --epochs 50 --batch-size 2 --bf16 --output-mode exp7 --no-nonneg-clamp

# accelerate launch train_hf_trainer_state_lb_cv.py --lb-state Vic --cv-folds 0,1,2,3,4 --output-mode exp6 --epochs 50 --batch-size 2 --bf16

# accelerate launch train_hf_trainer.py --folds 0,1,2,3,4 --epochs 50 --batch-size 2 --bf16 --output-mode exp1 --no-nonneg-clamp
# accelerate launch train_hf_trainer.py --folds 0,1,2,3,4 --epochs 50 --batch-size 2 --bf16 --output-mode exp2 --no-nonneg-clamp
# accelerate launch train_hf_trainer.py --folds 0,1,2,3,4 --epochs 50 --batch-size 2 --bf16 --output-mode exp3 --no-nonneg-clamp
# accelerate launch train_hf_trainer.py --folds 0,1,2,3,4 --epochs 50 --batch-size 2 --bf16 --output-mode exp5 --no-nonneg-clamp
