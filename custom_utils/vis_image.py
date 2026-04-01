from PIL import Image, ImageDraw, ImageFont, ImageColor

def wrap_text(text, font, max_width):
    lines = []
    current_line = ""
    for word in text.split():
        test_line = current_line + " " + word if current_line else word
        width = font.getbbox(test_line)[2] - font.getbbox(test_line)[0]
        if width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines

def draw_point(image: Image.Image, point: list, color=None, radius: int = 140):
    """
    在图像上绘制半透明圆点。
    外圈范围由 radius (即 norm 距离) 控制，默认为 140 像素。
    """
    # 1. 处理颜色：确保它是 RGBA 格式
    if isinstance(color, str):
        try:
            rgb = ImageColor.getrgb(color)
            # 如果是 RGB，加上透明度 128
            color = rgb + (128,) if len(rgb) == 3 else rgb
        except ValueError:
            color = (255, 0, 0, 128)
    elif isinstance(color, tuple) and len(color) == 3:
         color = color + (128,) # 如果传入的是 RGB 元组，补上透明度
    elif color is None:
        color = (255, 0, 0, 128)

    # 2. 创建用于绘制半透明层的 Overlay
    # 必须转换为 RGBA 才能进行 alpha_composite
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
        
    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    x, y = point

    # 3. 绘制外圈 (Norm 140 区域)
    # 这里的 radius 直接对应欧氏距离 140
    w, h = image.size 
    overlay_draw.ellipse(
        [(x - radius / 1000 * w, y - radius / 1000 * h), (x + radius / 1000 * w, y + radius / 1000 * h)],
        fill=color,
        outline=None # 如果只需要圈起来不要填充，可以把 fill 设为 None，设置 outline=color
    )
    
    # 4. 绘制中心绿色小点
    # 保持中心点大小相对外圈为 10% (即 14像素)，或者你可以将其改为固定大小
    center_radius = radius * 0.1 
    overlay_draw.ellipse(
        [(x - center_radius, y - center_radius), 
         (x + center_radius, y + center_radius)],
        fill=(0, 255, 0, 255)
    )

    # 5. 合并图像
    combined = Image.alpha_composite(image, overlay)
    
    # 根据需要决定是否转回 RGB (通常保存为 jpg 需要 RGB)
    return combined.convert('RGB')


def vis_func(org_image, user_text, pred, gt, padding=10, line_spacing=8):
    """
    可视化函数：
    - 显示原图 + bbox
    - 在 pred_point 位置绘制半透明圆点（外圈+中心点）
    """
    # === 1. 打开原图 ===
    if isinstance(org_image, str):
        image = Image.open(org_image).convert("RGB")
    else: image = org_image
    font = ImageFont.truetype("/mnt/vlm-ks3/zhangyan/datasets/font/timesbd.ttf", 24)

    # === 2. 拼接文本 ===
    combined_text = f"User text: {user_text}\n"
    max_text_width = image.width - 2 * padding

    def wrap_text(text, font, max_width):
        lines, line = [], ""
        for word in text:
            test_line = line + word
            w = font.getbbox(test_line)[2] - font.getbbox(test_line)[0]
            if w <= max_width:
                line = test_line
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)
        return lines

    all_lines = []
    for paragraph in combined_text.split("\n"):
        lines = wrap_text(paragraph, font, max_text_width)
        all_lines.extend(lines)

    # === 3. 文本区高度 ===
    line_height = font.getbbox("A")[3] - font.getbbox("A")[1] + line_spacing
    text_block_height = len(all_lines) * line_height + 2 * padding

    # === 4. 创建新画布 ===
    new_height = image.height + text_block_height + 2 * padding
    new_img = Image.new("RGB", (image.width, new_height), color=(255, 255, 255))
    new_img.paste(image, (0, 0))

    # === 5. 绘制 gt ===
    if gt != None:
        if gt == [0,0]:
            pass
        elif len(gt) == 4:
            draw = ImageDraw.Draw(new_img)
            draw.rectangle(gt, width=8, outline="green")
        else:
            new_img = draw_point(new_img, gt, color="green")

    # === 6. 绘制 pred ===
    if pred != None:
        if pred == [0,0]:
            pass
        elif len(pred) == 4:
            draw = ImageDraw.Draw(new_img)
            draw.rectangle(pred, width=8, outline="red")
        else:
            new_img = draw_point(new_img, pred, color="red")

    # === 7. 写文本说明 ===
    draw = ImageDraw.Draw(new_img)
    x, y = padding, image.height + padding
    for line in all_lines:
        draw.text((x, y), line, fill="black", font=font)
        y += line_height

    return new_img


def merge_vis_images(imgs, gap=20):   # gap 像素间距，默认 20，可自行调节
    # 1) 计算整张画布的宽度：所有图宽 + 间隔（数量 - 1） * gap
    w = sum(img.width for img in imgs) + gap * (len(imgs) - 1)
    h = max(img.height for img in imgs)

    canvas = Image.new("RGB", (w, h), color=(255, 255, 255))  # 背景白色，也可以改成黑色

    x = 0
    for img in imgs:
        canvas.paste(img, (x, 0))
        x += img.width + gap  # 每次向右偏移 width + gap

    return canvas