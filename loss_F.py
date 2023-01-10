import torch

def objectness_loss(input, target):
        sum_loss = 0
        for i in range(len(input)):
            sum_loss += (input[i,4]-target[i,4])**2
        return sum_loss
    
def obj_GIoU(input, target):
    total_loss = 0
    for i in range(len(input)):
        obj_loss = (input[i,4]-target[i,4])**2
        
        if target[i,4] == 0:
            box_loss = 0
        elif target[i,4] == 1:
            boxA = [input[i,0]-0.5*input[i,2],  input[i,1]-0.5*input[i,3],  input[i,0]+0.5*input[i,2],  input[i,1]+0.5*input[i,3]]
            boxB = [target[i,0]-0.5*target[i,2],target[i,1]-0.5*target[i,3],target[i,0]+0.5*target[i,2],target[i,1]+0.5*target[i,3]]
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            interArea = max(0, xB - xA) * max(0, yB - yA )
            All_area = (max(boxA[2],boxB[2])-min(boxA[0],boxB[0])) * (max(boxA[3],boxB[3]) - min(boxA[1], boxB[1]))
            C_box = (All_area - (boxAArea + boxBArea - interArea))/All_area
            iou = interArea / (boxAArea + boxBArea - interArea)
            if (All_area - (boxAArea + boxBArea - interArea)) < 0:
                GIoU = -1
    #            print('0101010101')
            elif min(input[i,:]) < 0:
                GIoU = -1       
            else:
                GIoU = iou - C_box
            if 1-GIoU > 2:
                print(boxA)
                print(boxB)
                print(input[i,:])
                print(min)
            if torch.is_tensor(GIoU) == False:
                GIoU = torch.cuda.FloatTensor([GIoU])
                GIoU = torch.autograd.Variable(GIoU, requires_grad=True)
                box_loss = (1-GIoU[0])
            else:
                box_loss = (1-GIoU)
        total_loss += box_loss + obj_loss
    return total_loss


    
def GIoU(input, target):
        sum_GIoUloss = 0
    #    input = input*1000
    #    target = target*1000
    #    target[:,0:2] = target[:,0:2]+1
    #    input[:,0:2] = input[:,0:2]+1
        for i in range(len(input)):
            boxA = [input[i,0]-0.5*input[i,2],  input[i,1]-0.5*input[i,3],  input[i,0]+0.5*input[i,2],  input[i,1]+0.5*input[i,3]]
            boxB = [target[i,0]-0.5*target[i,2],target[i,1]-0.5*target[i,3],target[i,0]+0.5*target[i,2],target[i,1]+0.5*target[i,3]]
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            interArea = max(0, xB - xA) * max(0, yB - yA )
            All_area = (max(boxA[2],boxB[2])-min(boxA[0],boxB[0])) * (max(boxA[3],boxB[3]) - min(boxA[1], boxB[1]))
            C_box = (All_area - (boxAArea + boxBArea - interArea))/All_area
            iou = interArea / (boxAArea + boxBArea - interArea)
            if (All_area - (boxAArea + boxBArea - interArea)) < 0:
                GIoU = -1
    #            print('0101010101')
            elif min(input[i,:]) < 0:
                GIoU = -1       
            else:
                GIoU = iou - C_box
            if 1-GIoU > 2:
                print(boxA)
                print(boxB)
                print(input[i,:])
                print(min)
            if torch.is_tensor(GIoU) == False:
                GIoU = torch.cuda.FloatTensor([GIoU])
                GIoU = torch.autograd.Variable(GIoU, requires_grad=True)
                sum_GIoUloss += (1-GIoU[0])
            else:
                sum_GIoUloss += (1-GIoU)
    
        return sum_GIoUloss
def IoU(input, target):
    sum_iou = 0
    target[:,0:2] = target[:,0:2]+1
    input[:,0:2] = input[:,0:2]+1
    for i in range(len(input)):
        boxA = [input[i,0]-0.5*input[i,2],  input[i,1]-0.5*input[i,3],  input[i,0]+0.5*input[i,2],  input[i,1]+0.5*input[i,3]]
        boxB = [target[i,0]-0.5*target[i,2],target[i,1]-0.5*target[i,3],target[i,0]+0.5*target[i,2],target[i,1]+0.5*target[i,3]]
    	# determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
    	# compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA )
        if interArea == 0:
            iou = 0
        elif min(input[i,:]) < 0:
            iou = 0
        else:
        	# compute the area of both the prediction and ground-truth
        	# rectangles
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        	# compute the intersection over union by taking the intersection
        	# area and dividing it by the sum of prediction + ground-truth
        	# areas - the interesection area
            iou = interArea / float(boxAArea + boxBArea - interArea)
        	# return the intersection over union value
        sum_iou = sum_iou + iou    
    return sum_iou

def negative_IoU(input, target):
    sum_iou = 0
    count = 0
    target[:,0:2] = target[:,0:2]+1
    input[:,0:2] = input[:,0:2]+1
    for i in range(len(input)):
        if target[i,4] == 1:
            boxA = [input[i,0]-0.5*input[i,2],  input[i,1]-0.5*input[i,3],  input[i,0]+0.5*input[i,2],  input[i,1]+0.5*input[i,3]]
            boxB = [target[i,0]-0.5*target[i,2],target[i,1]-0.5*target[i,3],target[i,0]+0.5*target[i,2],target[i,1]+0.5*target[i,3]]
        	# determine the (x, y)-coordinates of the intersection rectangle
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
        	# compute the area of intersection rectangle
            interArea = max(0, xB - xA) * max(0, yB - yA )
            if interArea == 0:
                iou = 0
            elif min(input[i,:]) < 0:
                iou = 0
            else:
            	# compute the area of both the prediction and ground-truth
            	# rectangles
                boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
                boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            	# compute the intersection over union by taking the intersection
            	# area and dividing it by the sum of prediction + ground-truth
            	# areas - the interesection area
                iou = interArea / float(boxAArea + boxBArea - interArea)
            	# return the intersection over union value
            sum_iou = sum_iou + iou    
            count += 1
    return sum_iou, count
