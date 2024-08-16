import os
import time
import random
import numpy as np
import pandas as pd
import scipy.stats as stats
from torch.cuda.amp import GradScaler,autocast
from collections import defaultdict, Counter
from calc_score import *
from torch import nn
import torch
import torchvision.transforms as transforms
import shutil
import inspect
from helpers import *
from Models.main_models_multi import Feature_Combination
from torch import optim
from opts_multi_modal import parse_opts
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader, Subset
from mri_pet_dataset_binary import MakeDataset as ImageDataset

args = parse_opts()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
cuda = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = args.random_seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True

scaler = GradScaler()

def copy_code(CODE_PATH):
    frame = inspect.currentframe().f_back
    caller_file = frame.f_code.co_filename
    rawabs_path = caller_file
    moveabs_path = '{}/{}'.format(CODE_PATH, caller_file.split('/')[-1])
    # moveabs_path =  '/home/sdb1/qujing/AD_Classifier/Classifier_save/{}/Code/{}/{}'.format(model_name, scale, caller_file.split('/')[-1])
    # os.makedirs(moveabs_path)
    shutil.copy(rawabs_path, moveabs_path)

def calculate_class_weights(dataset):
    labels = [label for _, label in dataset]
    class_counts = Counter(labels)
    total_samples = len(dataset)
    num_classes = len(class_counts)

    # Calculate weight for each class
    class_weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]
    return torch.tensor(class_weights, dtype=torch.float32)


# 训练函数
def train_class_model(dataloader, model, criterion, optimizer):
    model.train()
    running_loss = 0.0
    PREDICTED_val, REAL_val = [], []
    for images, labels in dataloader:
        train_batch_size = images.size()[0]

        mri_images = images[:, 0, :, :, :].view(train_batch_size, 1, args.Image_shape[0], args.Image_shape[1], args.Image_shape[2])
        pet_images = images[:, 1, :, :, :].view(train_batch_size, 1, args.Image_shape[0], args.Image_shape[1], args.Image_shape[2])

        mri_images = mri_images.to(device, dtype=torch.float32)  # .unsqueeze(1)
        pet_images = pet_images.to(device, dtype=torch.float32)  # .unsqueeze(1)
        labels = labels.to(device, dtype=torch.long)  # .unsqueeze(2)#.permute(0, 3, 1, 2)#

        for itera_f in range(args.iter_t):
            for p in model.parameters():
                p.requires_grad = True

            optimizer.zero_grad()
            # input
            outputs = model(mri_images, pet_images)
            # loss function
            loss = criterion(outputs, labels, sign=args.criterion_sign)

            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()

        # out_c = F.softmax(outputs, dim=1)
        if args.criterion_sign:
            # output = torch.add(torch.add(x5, y5), x_y_5)
            _, predicted_x = torch.max(outputs[0].data, 1)
            _, predicted_y = torch.max(outputs[1].data, 1)
            _, predicted_xy = torch.max(outputs[2].data, 1)
            predicted = torch.tensor([Counter([predicted_x[i], predicted_y[i], predicted_xy[i]]).most_common(1)[0][0] if Counter([predicted_x[i], predicted_y[i], predicted_xy[i]]).most_common(1) else 1 for i in range(len(predicted_x))])
        else:
            _, predicted = torch.max(outputs.data, 1)
        PREDICTED_ = predicted.data.cpu().numpy()
        REAL_ = labels.data.cpu().numpy()

        PREDICTED_val.extend(PREDICTED_)
        REAL_val.extend(REAL_)

        running_loss += loss.item() * labels.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)

    if args.task == 'class':
        print('Training PREDICTED', Counter(PREDICTED_val))
        print('Training REAL', Counter(REAL_val))

    metrics_to_compute = args.metrics
    metrics_multiclass_val = compute_metrics(PREDICTED_val, REAL_val, metrics_to_compute)
    return epoch_loss, metrics_multiclass_val

# 评估函数
def evaluate_class_model(dataloader, model, criterion):
    model.eval()
    running_loss = 0.0
    PREDICTED_test, REAL_test = [], []
    with torch.no_grad():
        for val_test_imgs, val_test_labels in dataloader:
            val_test_data_batch_size = val_test_imgs.size()[0]

            mri_images_ = val_test_imgs[:, 0, :, :, :].view(val_test_data_batch_size, 1, 76, 94, 76)
            pet_images_ = val_test_imgs[:, 1, :, :, :].view(val_test_data_batch_size, 1, 76, 94, 76)

            mri_images_ = mri_images_.to(device, dtype=torch.float32)  # .unsqueeze(1)
            pet_images_ = pet_images_.to(device, dtype=torch.float32)  # .unsqueeze(1)
            val_test_labels = val_test_labels.to(device, dtype=torch.long)  # .unsqueeze(2)#.permute(0, 3, 1, 2)#

            outputs = model(mri_images_, pet_images_)

            loss = criterion(outputs, val_test_labels, sign=args.criterion_sign) #criterion['ssim'](outputs, labels) +
            running_loss += loss.item() * val_test_labels.size(0)

            if args.criterion_sign:
                # output = torch.add(torch.add(x5, y5), x_y_5)
                _, predicted_x = torch.max(outputs[0].data, 1)
                _, predicted_y = torch.max(outputs[1].data, 1)
                _, predicted_xy = torch.max(outputs[2].data, 1)
                predicted = torch.tensor([Counter([predicted_x[i], predicted_y[i], predicted_xy[i]]).most_common(1)[0][0] for i in
                             range(len(predicted_x))])
            else:
                _, predicted = torch.max(outputs.data, 1)
            PREDICTED = predicted.data.cpu().numpy()
            REAL = val_test_labels.data.cpu().numpy()

            PREDICTED_test.extend(PREDICTED)
            REAL_test.extend(REAL)
        if args.task == 'class':
            print('Testing PREDICTED', Counter(PREDICTED_test))
            print('Testing REAL', Counter(REAL_test))
        metrics_to_compute = args.metrics  #['accuracy', 'precision', 'recall', 'F1-score', 'specificity', 'sensitivity', 'AUC']
        metrics_multiclass_test = compute_metrics(PREDICTED_test, REAL_test, metrics_to_compute)

    avg_loss = running_loss / len(dataloader.dataset)
    return avg_loss, metrics_multiclass_test


def write_setting_log(args, SETTING_PATH):
    f = open(os.path.join(SETTING_PATH, 'setting.log'), 'a')
    writelog(f, '======================')
    writelog(f, 'GPU ID: %s' % (args.gpu_id))
    writelog(f, 'Task: %s' % (args.task))
    writelog(f, 'Classes Num: %s' % (args.num_classes))
    writelog(f, 'Modal: %s' % (args.modal))
    writelog(f, 'Model Name: %s' % (args.model))
    writelog(f, '----------------------')
    writelog(f, 'Train Data Path: %s' % (args.train_path))
    writelog(f, 'Test Data Path: %s' % (args.test_path))
    writelog(f, 'Results Path: %s' % (args.save_path))
    writelog(f, '----------------------')
    writelog(f, 'Iter: %d' % args.n_iter)
    writelog(f, 'Fold: %d' % args.k_fold)
    writelog(f, 'Epoch: %d' % args.epoch)
    writelog(f, 'Train Batch Size: %d' % args.train_batch_size)
    writelog(f, 'Test Batch Size: %d' % args.test_batch_size)
    writelog(f, 'Learning Rate: %.5f' % args.lr)
    writelog(f, '----------------------')
    writelog(f, 'Model: %s' % args.model)
    writelog(f, 'criterion: %s' % args.criterion)
    writelog(f, 'optimizer: %s' % args.optimizer)
    writelog(f, 'Notes: %s' % args.notes)
    writelog(f, '======================')
    f.close()

def train_plane(scale, fold, EPOCH, loader_train, loader_val, model, criterion, optimizer, scheduler, LOG_PATH, MODEL_PATH):
    # Best epoch checking
    valid = {
        'epoch': 0,
        'loss': 0,
    }
    score_log = defaultdict(list)
    ES = EarlyStopping(delta=0, patience=20, verbose=True)

    f = open(os.path.join(LOG_PATH,'log_model_fold{}.log'.format(fold)), 'a')
    writelog(f, "----------------Fold {}: Training on {} plane---------------------".format(fold, plane))
    for epoch in range(EPOCH):
        start_time = time.time()
        writelog(f, '--- Epoch %d' %(epoch+1))
        # 训练和评估模型
        writelog(f, "Training ...")
        if args.task == 'class':
            epoch_loss, metrics_multiclass_train = train_class_model(loader_train, model, criterion, optimizer)
            loss_val, metrics_multiclass_test = evaluate_class_model(loader_val, model, criterion)  # evaluate('Validation', dataloader_valid)

            writelog(f, "multi-classification-results-Testing：")
            for metric, value in metrics_multiclass_test.items():
                writelog(f, f"{metric}: {value}")
            writelog(f, '\n')
            writelog(f, f'Epoch {epoch + 1}/{EPOCH}, Training Loss: {epoch_loss:.4f}, Validation Loss: {loss_val:.4f}')

        score_log['EPOCH'].append(epoch + 1)
        score_log['Task_Loss'].append(round(loss_val, 4))

        for metric in args.metrics:
            score_log['Train' + metric.upper()[:3]].append(metrics_multiclass_train[metric])
            score_log['Test'+metric.upper()[:3]].append(metrics_multiclass_test[metric])

        t_comp = (time.time() - start_time)
        score_log['Time_Taken'].append(round(t_comp, 4))

        if epoch == 0:
            valid['loss'] = loss_val

        # Save Model
        if loss_val < valid['loss']:
            torch.save(model.state_dict(),
                       os.path.join(MODEL_PATH, 'F{}_E{}_TLoss{}_T.pth'.format(fold, epoch + 1, round(loss_val, 4))))

            writelog(f, 'Saving model to {}'.format(
                os.path.join(MODEL_PATH, 'F{}_E{}_TLoss{}_T.pth'.format(fold, epoch + 1, round(loss_val, 4)))))
            writelog(f, 'Best validation loss is found! Validation loss : %f' % loss_val)
            writelog(f, 'Models at Fold %d Epoch %d are saved!' % (fold, epoch+1))

            valid['loss'] = loss_val
            valid['epoch'] = epoch+1
            ES(loss_val, None)

        scheduler.step()
        if ES.early_stop == True:
            break
    #
    writelog(f, 'END OF TRAINING')
    f.close()

    score_log_df = pd.DataFrame(score_log)
    LOG_SAVE = os.path.join(LOG_PATH,
                            "log_score_{}_{}_E{}_F{}.csv".format(args.model, scale, EPOCH, fold))
    score_log_df.to_csv(LOG_SAVE, index=False)

    if args.task == 'class':
        macro_scores = [item['macro'] for item in score_log_df['Test' + args.main_metric.upper()[:3]]]
        index_ = macro_scores.index(max(macro_scores))
        best_result = score_log_df.iloc[index_, :]
        return dict(best_result)
    else:
        return None

def main(args, fold, data_train, data_test):
    TRAIN_BATCH_SIZE = args.train_batch_size
    TEST_BATCH_SIZE = args.test_batch_size

    EPOCH = args.epoch
    LR = args.lr
    STEP_SIZE = args.scheduler_step
    GAMMA = args.scheduler_gamma
    LAST_EPOCH = args.scheduler_last_epoch

    loader_train = DataLoader(data_train, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    loader_test = DataLoader(data_test, batch_size=TEST_BATCH_SIZE, shuffle=False)

    class_weights = calculate_class_weights(data_train).cuda()

    model = nn.DataParallel(Feature_Combination(args)).to(device)
    model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA, last_epoch=LAST_EPOCH)

    best_performer = train_plane(scale, fold, EPOCH, loader_train, loader_test, model, criterion, optimizer,
                                 scheduler, LOG_PATH, MODEL_PATH)

    return best_performer

def calc_ci_error(predictions, confidence_level=0.95):
    degrees_freedom = len(predictions) - 1
    sample_mean = np.mean(predictions)
    sample_standard_error = stats.sem(predictions)
    confidence_interval = stats.t.interval(confidence_level, degrees_freedom, sample_mean, sample_standard_error)
    ci_error = confidence_interval[1] - sample_mean
    return f"{round(sample_mean, 4)} +- {round(ci_error, 4)}"


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])

    SAVE_DIR = args.save_path
    DATA_PATH = args.data_path
    DATA_TRAIN = args.train_path
    DATA_TEST = args.test_path

    TASK = args.task
    SEED = args.random_seed
    PLANE = args.plane
    IMAGE_SPACING = args.img_spacing
    N_ITER = args.n_iter
    K_FOLD = args.k_fold
    MODEL_NAME = args.model
    MODAL = args.modal
    results_path = os.path.join(os.path.join(SAVE_DIR, MODAL), MODEL_NAME)

    scale = time.asctime()
    SETTING_PATH = '{}/{}/Setting'.format(results_path, scale)
    LOG_PATH = '{}/{}/Log'.format(results_path, scale)
    CODE_PATH = '{}/{}/Code'.format(results_path, scale)
    MODEL_PATH = '{}/{}/Model'.format(results_path, scale)
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(CODE_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)
    os.makedirs(SETTING_PATH, exist_ok=True)

    best_performer = defaultdict(list)

    write_setting_log(args, SETTING_PATH)
    copy_code(CODE_PATH)
    shutil.copy(args.pwd, os.path.join(SETTING_PATH, args.pwd.split('/')[-1]))

    for t in range(N_ITER) :
        if K_FOLD:
            filepaths = os.listdir(DATA_PATH)
            All_labels = [name.split('_')[0] for name in os.listdir(DATA_PATH)]

            skf = StratifiedKFold(n_splits=K_FOLD, shuffle=True, random_state=SEED)

            for fold, (train_idx, val_idx) in enumerate(skf.split(filepaths, All_labels)):
                dataset = ImageDataset(DATA_PATH)
                train_subset = Subset(dataset, train_idx)
                test_subset = Subset(dataset, val_idx)

                i_best_performer = main(args, fold, train_subset, test_subset)
                for key, value in i_best_performer.items():
                    best_performer[key].append(value)

        else:
            data_train = ImageDataset(DATA_TRAIN)
            data_test = ImageDataset(DATA_TEST)

            i_best_performer = main(args, data_train, data_test)
            for key, value in i_best_performer.items():
                best_performer[key].append(value)
    res_df = pd.DataFrame(best_performer)
    len_row = res_df.shape[0]

    for col in res_df.columns[2:]:
        temp = defaultdict(list)
        for item in res_df[col]:
            if type(item) != dict:
                continue
            for key, value in item.items():
                temp[key].append(value)
        temp_mean = {key: calc_ci_error(values) for key, values in temp.items()}

        res_df.loc[len_row+1, col] = str(temp_mean)

    res_df.to_csv(os.path.join(LOG_PATH, '{}iter_{}fold_{}epoch_best_results.csv'.format(N_ITER, K_FOLD, args.epoch)), encoding='utf-8')






