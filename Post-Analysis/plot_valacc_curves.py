import os
import sys

sys.path.append(os.getcwd())

from post_anal_utils import extract_critical_role, pattern, plot_multiple_val_acc

val_acc_lr_decay_list1 = []
val_acc_lr_decay_list2 = []
val_acc_lr_decay_list3 = []

# DDETR
val_acc_lr_decay_list1.append((
    extract_critical_role('Deformable-DETR/exps/r50_deformable_detr/log_ddetr_nq75_e65.txt', pattern),
    (40, )
))
# DDETR no aux loss
val_acc_lr_decay_list1.append((
    extract_critical_role('Deformable-DETR/exps/r50_deformable_detr/log_ddetr_nq75_noauxloss.txt', pattern),
    (40, )
))
# # DDETR linear
# val_acc_lr_decay_list.append((
#     extract_critical_role('Deformable-DETR/exps/r50_deformable_detr/log_ddetr_nq75_lincoef_e65.txt', pattern),
#     (40, )
# ))
# DDETR fib
val_acc_lr_decay_list1.append((
    extract_critical_role('Deformable-DETR/exps/r50_deformable_detr/log_ddetr_nq75_fibcoef_e65.txt', pattern),
    (40, )
))
# SQR DDETR s13
val_acc_lr_decay_list2.append((
    extract_critical_role('Deformable-DETR/exps/r50_deformable_detr/log_sqrddetr_nq75_s13_e65.txt', pattern),
    (40, )
))
# SQR DDETR s20
val_acc_lr_decay_list2.append((
    extract_critical_role('Deformable-DETR/exps/r50_deformable_detr/log_sqrddetr_nq75_s20_e65.txt', pattern),
    (40, )
))
# SQR DDETR s32
val_acc_lr_decay_list2.append((
    extract_critical_role('Deformable-DETR/exps/r50_deformable_detr/log_sqrddetr_nq75_s32_e65.txt', pattern),
    (40, )
))
# SQR DDETR s32
val_acc_lr_decay_list1.append((
    extract_critical_role('Deformable-DETR/exps/r50_deformable_detr/log_sqrddetr_nq75_s32_e65.txt', pattern),
    (40, )
))
# SQR DDETR s32-fib
val_acc_lr_decay_list1.append((
    extract_critical_role('Deformable-DETR/exps/r50_deformable_detr/log_sqrddetr_nq75_fibcoef_e65.txt', pattern),
    (40, )
))

# DDETR no aux loss
val_acc_lr_decay_list3.append((
    extract_critical_role('Deformable-DETR/exps/r50_deformable_detr/log_ddetr_nq75_noauxloss.txt', pattern),
    (40, )
))
# DDETR linear
val_acc_lr_decay_list3.append((
    extract_critical_role('Deformable-DETR/exps/r50_deformable_detr/log_ddetr_nq75_lincoef_e65.txt', pattern),
    (40, )
))
# DDETR fib
val_acc_lr_decay_list3.append((
    extract_critical_role('Deformable-DETR/exps/r50_deformable_detr/log_ddetr_nq75_fibcoef_e65.txt', pattern),
    (40, )
))

# plot here.
plot_multiple_val_acc(val_acc_lr_decay_list1, fname='valaccs1.png',
                      legends=['DDETR', 'DDETR-wo-auxloss', 'DDETR-fib', 'SQRDDETR-s32', 'SQRDDETR-s32-fib'])

plot_multiple_val_acc(val_acc_lr_decay_list2, fname='valaccs2.png',
                      legends=['SQRDDETR-s13', 'SQRDDETR-s20', 'SQRDDETR-s32', 'SQRDDETR-s32-fib'])

plot_multiple_val_acc(val_acc_lr_decay_list3, fname='valaccs3.png',
                      legends=['DDETR-wo-auxloss', 'DDETR-linear', 'DDETR-fib'])
