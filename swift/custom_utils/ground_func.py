import os 
import math 

def get_scroll_direction(start, end):
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1

    if abs(dy) > abs(dx):
        return "up" if dy < 0 else "down"
    else:
        return "left" if dx < 0 else "right"

def bboxnorm2real(norm_bbox, image_size):
    width, height =  image_size
    bbox = [
        int(norm_bbox[0] / 1000 * width),
        int(norm_bbox[1] / 1000 * height),
        int(norm_bbox[2] / 1000 * width),
        int(norm_bbox[3] / 1000 * height)
    ]
    return bbox

def bboxreal2norm(real_bbox, image_size):
    width, height =  image_size
    bbox = [
        int(real_bbox[0] / width * 1000),
        int(real_bbox[1] / height * 1000),
        int(real_bbox[2] / width * 1000),
        int(real_bbox[3] / height * 1000)
    ]
    return bbox
    
def pointnorm2real(norm_point, image_size):
    width, height =  image_size
    point = [
        int(norm_point[0] / 1000 * width),
        int(norm_point[1] / 1000 * height),
    ]
    return point

def pointreal2norm(real_point, image_size):
    width, height =  image_size
    point = [
        int(real_point[0] /width * 1000),
        int(real_point[1] /height * 1000),
    ]
    return point


# qwen2.5vl 使用
def qwen25_get_resize(img_size):
    from qwen_vl_utils import smart_resize
    width, height = img_size
    max_pixels = int(os.getenv("MAX_PIXELS",4390400))
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=28,
        min_pixels=3136,
        max_pixels=max_pixels,
    )
    return resized_width, resized_height


def calculate_pred_norm_point(image_size, pred_point, model_name):
    if model_name == "qwen25vl":
        # pred for qwen2.5vl
        resized_size = qwen25_get_resize(image_size)
        pred_x, pred_y = pointreal2norm(pred_point, resized_size)

    elif model_name == "qwen3vl":
        # pred for qwen3vl
        pred_x, pred_y = pred_point

    return pred_x, pred_y 


def ground_reward_jigsaw_func(
    pred_point,
    ground_truth,
    image_size,
    ground_reward,
    hier_click_reward_score,
):
    pred_x, pred_y = pred_point 
    
    # jigsaw 后的 patch_bbox
    patch_bbox = ground_truth['jigsaw_info']['point']['patch_bbox']
    norm_patch_bbox = bboxreal2norm(patch_bbox, image_size)

    # point reward 
    if ground_reward == "point":
        gt_point = ground_truth["arguments"].get("coordinate")
        gt_x, gt_y = pointreal2norm(gt_point, image_size)
        
        if (pred_x-gt_x)**2+(pred_y-gt_y)**2<140**2:
            return 1.0

        if norm_patch_bbox[0] <= pred_x <= norm_patch_bbox[2] and \
            norm_patch_bbox[1] <= pred_y <= norm_patch_bbox[3]: 
            return hier_click_reward_score 
        else:
            return 0.0 

    elif ground_reward == "bbox":
        gt_bbox = bboxreal2norm(ground_truth['bbox'], image_size)
        
        

        if gt_bbox[0] <= pred_x <= gt_bbox[2] and gt_bbox[1] <= pred_y <= gt_bbox[3]:
            return 1.0
        
       
        
    else:
        print("error in calculate ground reward")
        return 0.0 

def g2_point_reward(pred_bbox, gt_bbox):
    alpha = 0.5
    pred_center_x, pred_center_y = pred_bbox
    gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox
    
    # 计算中心点
    gt_center_x = (gt_x1 + gt_x2) / 2
    gt_center_y = (gt_y1 + gt_y2) / 2
    gt_width = gt_x2 - gt_x1
    gt_height = gt_y2 - gt_y1
    
    sigma_x = alpha * gt_width
    sigma_y = alpha * gt_height

    x_term = (pred_center_x - gt_center_x)**2 / (sigma_x**2)
    y_term = (pred_center_y - gt_center_y)**2 / (sigma_y**2)
    exponent = -0.5 * (x_term + y_term)
    point_reward = math.exp(exponent)
    point_reward = round(point_reward,3)
    return point_reward
    
def ground_reward_func(
    pred_point,
    ground_truth,
    image_size,
):
    pred_x, pred_y = pred_point 
    gt_bbox = bboxreal2norm(ground_truth['arguments']['coordinate'], image_size)
    
    if gt_bbox[0] <= pred_x <= gt_bbox[2] and gt_bbox[1] <= pred_y <= gt_bbox[3]:
        return 1.0
    else:
        return 0.0
        