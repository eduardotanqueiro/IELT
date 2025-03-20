from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import AverageMeter, accuracy
from tqdm import tqdm

from models.build import build_models, freeze_backbone
from setup import config, log
from utils.data_loader import build_loader
from utils.eval import *
from utils.info import *
from utils.optimizer import build_optimizer
from utils.scheduler import build_scheduler

import wandb
import optuna
from utils.data_loader import new_datasets
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

import gc

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def build_model(config, num_classes):
	model = build_models(config, num_classes)
	if torch.__version__[0] == 2:
		model = torch.compile(model, mode="max-autotune")
	model.cuda()
	freeze_backbone(model, config.train.freeze_backbone)
	model_without_ddp = model
	n_parameters = count_parameters(model)

	config.defrost()
	config.model.num_classes = num_classes
	config.model.parameters = f'{n_parameters:.3f}M'
	config.freeze()
	if config.local_rank in [-1, 0]:
		PSetting(log, 'Model Structure', config.model.keys(), config.model.values(), rank=config.local_rank)
		log.save(model)
	return model, model_without_ddp

def cm_plot(y_true, y_preds , labels = None, save_path = None, plotit = True, per_class = False):

    cm = confusion_matrix(y_true = y_true, y_pred = y_preds, labels=labels)

    fig, ax = plt.subplots(figsize=(11, 10))
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    
    if save_path:
        fig.savefig(save_path)
    
    if plotit:
        plt.show()

    # plt.close(fig)

    # Calculate accuracy
    acc = accuracy_score(y_true, y_preds)

    # Calculate per-class recall (sensitivity), precision, and specificity
    recall_per_class = recall_score(y_true, y_preds, average=None, zero_division=0.0, labels=labels)
    precision_per_class = precision_score(y_true, y_preds, average=None, zero_division=0.0, labels=labels)

    # Calculate F1 score per class
    f1_per_class = []
    for prec_class, rec_class in zip(precision_per_class, recall_per_class):
        if prec_class + rec_class == 0:
            f1_per_class.append(0)
        else:
            f1_per_class.append(2 * (prec_class * rec_class) / (prec_class + rec_class))


    # Calculate specificity manually
    specificity_per_class = []
    for i in range(len(cm)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])  # TN for class i
        fp = cm[:, i].sum() - cm[i, i]                                # FP for class i
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0  # Avoid division by zero
        specificity_per_class.append(specificity)

    # Calculate macro-averaged metrics
    recall = np.mean(recall_per_class)
    prec = np.mean(precision_per_class)
    spec = np.mean(specificity_per_class)
    f1 = np.mean(f1_per_class)

    if per_class:
        return acc* 100. , recall* 100. , spec* 100. , prec* 100., f1*100, recall_per_class, precision_per_class, specificity_per_class, f1_per_class ,fig
    
    return acc* 100. , recall* 100. , spec* 100. , prec* 100., f1*100, fig


def objective(trial):
	# Timer
	prepare_timer = Timer()
	prepare_timer.start()
	train_timer = Timer()
	eval_timer = Timer()

	# Initialize the Tensorboard Writer
	# torch.cuda.empty_cache()
	# torch.cuda.init()
	# ---- #
	print(f"CUDA AVAILABLE: {torch.cuda.is_available()}")

	# ???
	# weights = np.load("E:\Thesis\IELT\pretrained\ViT-B_16.npz")
	
	# for k in weights.keys():
	# 	print(f"{k} || {weights[k].shape}")


	# exit(0)

	# ---- #

	# Clear Memory	

	model = None

	print(f"allocated {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
	print(f"reserved {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

	torch.cuda.empty_cache()
	gc.collect()

	print(f"allocated {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
	print(f"reserved {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

	# --- Optuna Hyperparameter Optimization ---
	config.defrost()

	# Dataset
	config.data.data_root = trial.suggest_categorical("data_root", new_datasets.keys())

	# Learning Rate
	config.train.lr = trial.suggest_float("lr", 5e-5, 5e-3, log=True)

	# Batch Size
	config.data.batch_size = trial.suggest_categorical("batch_", [8, 16])

	run_id = None
	with open("last_model_id.txt", "r+") as f:
		run_id = int(f.read())
		run_id += 1

		f.seek(0)
		f.write(str(run_id))

	if run_id is None:
		print("Error: Run ID not found")

	# Prepare dataset
	train_loader, test_loader, num_classes, train_samples, test_samples, mixup_fn = build_loader(config)
	step_per_epoch = len(train_loader)
	total_batch_size = config.data.batch_size * get_world_size()
	steps = config.train.epochs * step_per_epoch

	# Create config for WandB
	config_wandb = dict(
        run_id = run_id,
        epochs = config.train.epochs,
        classes = num_classes,
        batch_size = config.data.batch_size,
        learning_rate = config.train.lr,
        architecture = "IELT",
        dataset_version = config.data.data_root,
        num_train = len(train_loader.dataset),
        num_val = len(test_loader.dataset),
        img_size = config.data.img_size,
    )

	# Build model
	model, model_without_ddp = build_model(config, num_classes)
	if not config.model.baseline_model:
		model.encoder.warm_steps = config.parameters.update_warm * step_per_epoch
	if config.local_rank != -1:
		model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.local_rank],
		                                                  broadcast_buffers=False,
		                                                  find_unused_parameters=False)
	optimizer = build_optimizer(config, model, backbone_low_lr=False)
	loss_scaler = NativeScalerWithGradNormCount()
	# Build learning rate scheduler

	if config.train.lr_epoch_update:
		scheduler = build_scheduler(config, optimizer, 1)
	else:
		scheduler = build_scheduler(config, optimizer, step_per_epoch)

	# Determine criterion
	best_acc, best_epoch, train_accuracy = 0., 0., 0.

	# if config.data.mixup > 0.:
	# 	criterion = SoftTargetCrossEntropy()
	# elif config.model.label_smooth:
	# 	criterion = LabelSmoothingCrossEntropy(smoothing=config.model.label_smooth)
	# else:
	criterion = torch.nn.CrossEntropyLoss()

	# Function Mode
	if config.model.resume:
		best_acc = load_checkpoint(config, model_without_ddp, optimizer, scheduler, loss_scaler, log)
		best_epoch = config.start_epoch
		accuracy, loss = valid(config, model, test_loader, best_epoch, train_accuracy)
		log.info(f'Epoch {best_epoch:^3}/{config.train.epochs:^3}: Accuracy {accuracy:2.3f}    '
		         f'BA {best_acc:2.3f}    BE {best_epoch:3}    '
		         f'Loss {loss:1.4f}    TA {train_accuracy * 100:2.2f}')
		if config.misc.eval_mode:
			return

	if config.misc.throughput:
		throughput(test_loader, model, log, config.local_rank)
		return

	# Record result in Markdown Table
	mark_table = PMarkdownTable(log, ['Epoch', 'Accuracy', 'Best Accuracy',
	                                  'Best Epoch', 'Loss'], rank=config.local_rank)

	# End preparation
	torch.cuda.synchronize()
	prepare_time = prepare_timer.stop()
	PSetting(log, 'Training Information',
	         ['Train samples', 'Test samples', 'Total Batch Size', 'Load Time', 'Train Steps',
	          'Warm Epochs'],
	         [train_samples, test_samples, total_batch_size,
	          f'{prepare_time:.0f}s', steps, config.train.warmup_epochs],
	         newline=2, rank=config.local_rank)

	# Train Function

	# with wandb.init(project="ViT-Pretrained-Demo", config = config_wandb, name= f"{run_id}-IELT") as run:
	with wandb.init(project="Plant-Disease-Detection", config = config_wandb, name= f"{run_id}-IELT") as run:

		# Log Best Metrics
		best_stats = {"epoch": 0, "acc": 0.0, "spec": 0.0, "recall": 0.0, "prec": 0.0, "f1": 0.0, "loss": 0.0, "inf_time": 0.0, "conf": 0.0}

		wandb.watch(model, optimizer, log= "all", idx = run_id)

		sub_title(log, 'Start Training', rank=config.local_rank)
		for epoch in range(config.train.start_epoch, config.train.epochs):

			train_timer.start()
			if config.local_rank != -1:
				train_loader.sampler.set_epoch(epoch)
			
			if not config.misc.eval_mode:
				train_loss, train_accuracy = train_one_epoch(config, model, criterion, train_loader, optimizer,
												epoch, scheduler, loss_scaler, mixup_fn)
			
			train_timer.stop()

			# Eval Function
			eval_timer.start()
			if epoch < 5 or (epoch + 1) % config.misc.eval_every == 0 or epoch + 1 == config.train.epochs:

				accuracy, loss, y_preds, y_label, avg_confidence, inf_time = valid(config, model, test_loader, epoch, train_accuracy)

				acc, recall, spec, prec, f1, recall_per_class, spec_per_class, prec_per_class, f1_per_class, fig  = cm_plot(y_label, y_preds, labels=[i for i in range(num_classes)] ,per_class=True ,plotit=False)
				
				print(f"[TRAIN] - Loss: {train_loss:.3f}, Accuracy {train_accuracy:.3f}")
				print(f"[VALIDATION] - Loss:{loss:.3f}, Accuracy:{acc:.3f}, Recall:{recall:.3f},:{spec:.3f}, Precision:{prec:.3f}, F1:{f1:.3f}, Inference Time:{inf_time:.7f} s")
	
				# Log metrics
				wandb.log({ 
						"train_loss": train_loss, 
						"train_accuracy": train_accuracy, 
						"val_accuracy": acc,  
						"val_recall": recall, 
						"val_specificity": spec, 
						"val_precision": prec,
						"val_f1": f1, 
						"val_loss": loss, 
						"val_avg_confidence": avg_confidence,
						"val_inference_time": inf_time,
						"lr": optimizer.param_groups[0]['lr']})

				# Optuna Logging
				trial.report(acc, epoch)

				# Update Best Stats
				if acc > best_stats["acc"]:
					best_stats["epoch"] = epoch
					best_stats["acc"] = acc
					best_stats["spec"] = spec
					best_stats["recall"] = recall
					best_stats["prec"] = prec
					best_stats["f1"] = f1
					best_stats["loss"] = loss
					best_stats["inf_time"] = inf_time
					best_stats["conf"] = avg_confidence

				# TODO Early Stopping???
				if epoch >= 1 and epoch > best_stats["epoch"] + 5 and acc < best_stats["acc"]:
					print(f"Early stopping on epoch {epoch+1}")
					break
				
				if trial.should_prune():			
					wandb.log({"Pruned": True})
					wandb.finish()
					raise optuna.exceptions.TrialPruned()

				wandb.log({"Confusion Matrix": wandb.Image(fig)})

				fig, ax = plt.subplots()
				ax.bar(range(num_classes), recall_per_class, label='Recall', color='blue')
				ax.set_xlabel('Class Number')
				ax.set_ylabel('Recall')
				wandb.log({"Recall per class": fig})

				fig, ax = plt.subplots()
				ax.bar(range(num_classes), spec_per_class, label='Specificity', color='red')
				ax.set_xlabel('Class Number')
				ax.set_ylabel('Specificity')
				wandb.log({"Specificity per class": fig})

				fig, ax = plt.subplots()
				ax.bar(range(num_classes), prec_per_class, label='Precision', color='green')
				ax.set_xlabel('Class Number')
				ax.set_ylabel('Precision')
				wandb.log({"Precision per class": fig})

				fig, ax = plt.subplots()
				ax.bar(range(num_classes), f1_per_class, label='F1', color='purple')
				ax.set_xlabel('Class Number')
				ax.set_ylabel('F1-Score')
				wandb.log({"F1 per class": fig})

				plt.close('all')


				if config.local_rank in [-1, 0]:
					if best_acc < accuracy:
						best_acc = accuracy
						best_epoch = epoch + 1
						if config.write and epoch > 10 and config.train.checkpoint:
							save_checkpoint(config, epoch, model, best_acc, optimizer, scheduler, loss_scaler, log)
					log.info(f'Epoch {epoch + 1:^3}/{config.train.epochs:^3}: Accuracy {accuracy:2.3f}    '
							f'BA {best_acc:2.3f}    BE {best_epoch:3}    '
							f'Loss {loss:1.4f}    TA {train_accuracy * 100:2.2f}')
					if config.write:
						mark_table.add(log, [epoch + 1, f'{accuracy:2.3f}',
											f'{best_acc:2.3f}', best_epoch, f'{loss:1.5f}'], rank=config.local_rank)
				pass  # Eval
	
			eval_timer.stop()
			pass  # Train

		# Finish Training	
		# Log the final statistics as an individual log
		wandb.log(best_stats)

		train_time = train_timer.sum / 60
		eval_time = eval_timer.sum / 60
		total_time = train_time + eval_time
		PSetting(log, "Finish Training",
				['Best Accuracy', 'Best Epoch', 'Training Time', 'Testing Time', 'Total Time'],
				[f'{best_acc:2.3f}', best_epoch, f'{train_time:.2f} min', f'{eval_time:.2f} min', f'{total_time:.2f} min'],
				newline=2, rank=config.local_rank)
	
	return best_stats["acc"]


def train_one_epoch(config, model, criterion, train_loader, optimizer, epoch, scheduler, loss_scaler,
                    mixup_fn=None):
	model.train()
	optimizer.zero_grad()

	step_per_epoch = len(train_loader)
	loss_meter = AverageMeter()
	norm_meter = AverageMeter()
	scaler_meter = AverageMeter()
	epochs = config.train.epochs
	p_bar = tqdm(total=step_per_epoch,
	             desc=f'Train {epoch + 1:^3}/{epochs:^3}',
	             dynamic_ncols=True,
	             ascii=True,
	             disable=config.local_rank not in [-1, 0])
	
	all_preds, all_label = None, None
	
	for step, (x, y) in enumerate(train_loader):
		global_step = epoch * step_per_epoch + step
		x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
		if mixup_fn:
			x, y = mixup_fn(x, y)
		with torch.amp.autocast("cuda",enabled=config.misc.amp):
			if config.model.baseline_model:
				logits = model(x)
			else:
				logits = model(x, y)

		logits, loss = loss_in_iters(logits, y, criterion)

		is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
		grad_norm = loss_scaler(loss, optimizer, clip_grad=config.train.clip_grad,
		                        parameters=model.parameters(), create_graph=is_second_order)

		optimizer.zero_grad()
		if config.train.lr_epoch_update:
			scheduler.step_update(epoch + 1)
		else:
			scheduler.step_update(global_step + 1)
		loss_scale_value = loss_scaler.state_dict()["scale"]

		if mixup_fn is None:
			preds = torch.argmax(logits, dim=-1)
			all_preds, all_label = save_preds(preds, y, all_preds, all_label)
		torch.cuda.synchronize()

		# TODO CHECK IF PREDS ARE BEING UPDATED

		if grad_norm is not None:
			norm_meter.update(grad_norm)
		scaler_meter.update(loss_scale_value)
		loss_meter.update(loss.item(), y.size(0))

		lr = optimizer.param_groups[0]['lr']

		# set_postfix require dic input
		p_bar.set_postfix(loss="%2.5f" % loss_meter.avg, lr="%.5f" % lr, gn="%1.4f" % norm_meter.avg)
		# print(f"loss {loss_meter.avg}")

		p_bar.update()

	# After Training an Epoch
	p_bar.close()
	train_accuracy = eval_accuracy(all_preds, all_label, config) if mixup_fn is None else 0.0
	return loss_meter.avg , train_accuracy


def loss_in_iters(output, targets, criterion):
	if not isinstance(output, (list, tuple)):
		return output, criterion(output, targets)
	else:
		logits, loss = output
		return logits, loss


@torch.no_grad()
def valid(config, model, test_loader, epoch=-1, train_acc=0.0):
	criterion = torch.nn.CrossEntropyLoss()
	model.eval()

	step_per_epoch = len(test_loader)
	p_bar = tqdm(total=step_per_epoch,
	             desc=f'Valid {(epoch + 1) // config.misc.eval_every:^3}/{math.ceil(config.train.epochs / config.misc.eval_every):^3}',
	             dynamic_ncols=True,
	             ascii=True,
	             disable=config.local_rank not in [-1, 0])

	loss_meter = AverageMeter()
	acc_meter = AverageMeter()
	confidence_meter = AverageMeter()

	predictions = []
	labels = []

    # start timer
	st = time.perf_counter()

	for step, (x, y) in enumerate(test_loader):
		x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

		with torch.amp.autocast("cuda", enabled=config.misc.amp):
			logits = model(x)

		logits, loss = loss_in_iters(logits, y, criterion)

		acc = accuracy(logits, y)[0]
		if config.local_rank != -1:
			acc = reduce_mean(acc)

		loss_meter.update(loss.item(), y.size(0))
		acc_meter.update(acc.item(), y.size(0))
		confidence_meter.update(logits.softmax(dim=1).max(dim=1)[0].mean().item(), y.size(0))

		p_bar.set_postfix(acc="{:2.3f}".format(acc_meter.avg), loss="%2.5f" % loss_meter.avg,
		                  tra="{:2.3f}".format(train_acc * 100), conf="%2.3f" % (confidence_meter.avg * 100) )
		p_bar.update()

		predictions.extend(logits.argmax(dim=1).cpu().numpy())
		labels.extend(y.cpu().numpy())

		pass

	# end timer
	et = time.perf_counter()

	p_bar.close()


	return acc_meter.avg, loss_meter.avg, predictions, labels, confidence_meter.avg * 100, (et - st) / len(test_loader.dataset)


@torch.no_grad()
def throughput(data_loader, model, log, rank):
	model.eval()
	for idx, (images, _) in enumerate(data_loader):
		images = images.cuda(non_blocking=True)
		batch_size = images.shape[0]
		for i in range(50):
			model(images)
		torch.cuda.synchronize()
		if rank in [-1, 0]:
			log.info(f"throughput averaged with 30 times")
		tic1 = time.time()
		for i in range(30):
			model(images)
		torch.cuda.synchronize()
		tic2 = time.time()
		if rank in [-1, 0]:
			log.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
		return


if __name__ == '__main__':
	
	# Optuna Storage
    os.makedirs( os.path.join(".", "optuna_db"), exist_ok=True)
    storage = f"sqlite:///optuna_db/IELT.db"

    study = optuna.create_study(
        study_name="IELT",
        storage=storage,
        load_if_exists=True,
        direction="maximize", 
        pruner=optuna.pruners.MedianPruner())
    
    hours = 0.5
    # study.optimize(objective, n_trials=50, show_progress_bar=True)
    study.optimize(objective, timeout = 3600*hours, show_progress_bar=True)

    print(f"Best trial params: {study.best_params}")
