# Given group allocation and outcomes, this func gives the outcomes of control and trtments allcoated ito those
# groups so fat
# Output is a list of list
# control = []
# trt = []
# if group[0] == 0:
#     control.append([outcome[0]])
# else:
#     trt.append([outcome[0]])
#
# for i in range(1, len(group)):
#     if group[i] == 0:
#         control.append(control[-1]+[outcome[i]])
#         trt.append(trt[-1])
#     else:
#         control.append(control[-1])
#         trt.append(trt[-1]+[outcome[i]])
#
# c_outcome = [outcome[i] for i in range(len(outcome)) if group[i]==0]
# t_outcome = [outcome[i] for i in range(len(outcome)) if group[i]==1]
#