import os
import torch
from tqdm import tqdm

from utils.utilss import get_lr


# def fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen,
#                   gen_val, Epoch, cuda):
#     loss = 0
#     val_loss = 0
#
#     model_train.train()
#     print('Start Train')
#     with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
#         for iteration, batch in enumerate(gen):
#             if iteration >= epoch_step:
#                 break
#
#             images, targets = batch[0], batch[1]
#             with torch.no_grad():
#                 if cuda:
#                     images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
#                     targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
#                 else:
#                     images = torch.from_numpy(images).type(torch.FloatTensor)
#                     targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
#             # ----------------------#
#             #   清零梯度
#             # ----------------------#
#             optimizer.zero_grad()
#             # ----------------------#
#             #   前向传播
#             # ----------------------#
#             outputs = model_train(images)
#
#             # ----------------------#
#             #   计算损失
#             # ----------------------#
#             loss_value = yolo_loss(outputs, targets)
#
#             # ----------------------#
#             #   反向传播
#             # ----------------------#
#             loss_value.backward()
#             optimizer.step()
#
#             loss += loss_value.item()
#
#             pbar.set_postfix(**{'loss': loss / (iteration + 1),
#                                 'lr': get_lr(optimizer)})
#             pbar.update(1)
#
#     print('Finish Train')
#
#     model_train.eval()
#     print('Start Validation')
#     with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
#         for iteration, batch in enumerate(gen_val):
#             if iteration >= epoch_step_val:
#                 break
#             images, targets = batch[0], batch[1]
#             with torch.no_grad():
#                 if cuda:
#                     images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
#                     targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
#                 else:
#                     images = torch.from_numpy(images).type(torch.FloatTensor)
#                     targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
#                 # ----------------------#
#                 #   清零梯度
#                 # ----------------------#
#                 optimizer.zero_grad()
#                 # ----------------------#
#                 #   前向传播
#                 # ----------------------#
#                 outputs = model_train(images)
#
#                 # ----------------------#
#                 #   计算损失
#                 # ----------------------#
#                 loss_value = yolo_loss(outputs, targets)
#
#             val_loss += loss_value.item()
#             pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
#             pbar.update(1)
#
#     print('Finish Validation')
#
#     loss_history.append_loss(loss / epoch_step, val_loss / epoch_step_val)
#     print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
#     print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
#     if ((epoch + 1) % 10 == 0 and (epoch + 1) > 10):
#         torch.save(model.state_dict(),
#                    'logs/ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val))

def fit_one_epoch(model_train, model, fcos_loss, loss_history, eval_callback, optimizer, epoch, epoch_step,
                  epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0, best_map=0):
    total_loss  = 0
    val_loss    = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, bboxes, classes = batch[0], batch[1],  batch[2]
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                bboxes  = bboxes.cuda(local_rank)
                classes = classes.cuda(local_rank)
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            #-------------------#
            #   获得预测结果
            #-------------------#
            outputs = model_train(images)
            #-------------------#
            #   计算损失
            #-------------------#
            loss = fcos_loss(outputs, bboxes, classes)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #-------------------#
                #   获得预测结果
                #-------------------#
                outputs = model_train(images)
                #-------------------#
                #   计算损失
                #-------------------#
                loss = fcos_loss(outputs, bboxes, classes)

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()

        if local_rank == 0:
            pbar.set_postfix(**{'loss'  : total_loss / (iteration + 1),
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, bboxes, classes = batch[0], batch[1],  batch[2]
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                bboxes  = bboxes.cuda(local_rank)
                classes = classes.cuda(local_rank)

            optimizer.zero_grad()
            outputs     = model_train(images)
            loss        = fcos_loss(outputs, bboxes, classes)
            val_loss    += loss.item()
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        mAP = eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        # if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        #     torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-mAP-%.3f.pth' % (epoch + 1, total_loss / epoch_step, mAP)))

        if mAP > best_map:
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "epoch_%03d_weights_mAP_%.3f.pth" % (epoch + 1, mAP)))
            best_map = mAP
    # 把最好的结果返回了，否则的话修改不了这个值
    return best_map, mAP
        # torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))