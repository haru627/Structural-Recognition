import pathlib
import cv2
import numpy as np
import pandas as pd
import math



def detect_shapes(image):
    blurred = cv2.GaussianBlur(image, (1, 1), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 赤領域のマスク
    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.add(red_mask1, red_mask2)

    # 青領域のマスク
    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # GaussianBlurでノイズ除去
    red_mask = cv2.GaussianBlur(red_mask, (1, 1), 0)
    blue_mask = cv2.GaussianBlur(blue_mask, (1, 1), 0)

    # 輪郭の検出
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours_red, contours_blue, red_mask, blue_mask


def get_quadrilateral_info(contours):
    quadrilaterals = []
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 500:
            side_lengths = [np.linalg.norm(approx[i] - approx[(i + 1) % 4]) for i in range(4)]
            if all(length > 50 for length in side_lengths):
                quadrilaterals.append(approx)
    return quadrilaterals

def filter_small_quadrilaterals(quadrilaterals, mask, min_area=1000, min_pixels=500):

    filtered_quads = []
    for quad in quadrilaterals:
        area = cv2.contourArea(quad)
        # マスクを使用してピクセル数を計算
        temp_mask = np.zeros_like(mask)
        cv2.fillPoly(temp_mask, [quad], 255)
        pixel_count = cv2.countNonZero(temp_mask)
        
        # 面積とピクセル数でフィルタリング
        if area >= min_area and pixel_count >= min_pixels:
            filtered_quads.append(quad)
    return filtered_quads


def calculate_lengths_angles(quadrilateral, mask):
    points = quadrilateral.reshape(4, 2)
    lengths = [np.linalg.norm(points[i] - points[(i + 1) % 4]) for i in range(4)]

    # 最長の辺を探し、その辺が `AB` になるように頂点を並び替え
    max_length_idx = np.argmax(lengths)
    points = np.roll(points, -max_length_idx, axis=0)

    lengths, angles = [], []
    for i in range(4):
        pt1, pt2 = points[i], points[(i + 1) % 4]
        lengths.append(np.linalg.norm(pt1 - pt2))

    for i in range(4):
        pt_prev = points[i - 1]  # 頂点iの1つ前 (反時計回りの隣接点)
        pt_current = points[i]  # 現在の頂点
        pt_next = points[(i + 1) % 4]  # 頂点iの次 (反時計回りの隣接点)

        # ベクトル計算
        v1 = pt_prev - pt_current
        v2 = pt_next - pt_current

        # 内積と角度計算
        angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
        angles.append(np.degrees(angle))

    # 総ピクセル数の計算
    mask_filled = np.zeros_like(mask)
    cv2.fillPoly(mask_filled, [quadrilateral], 255)
    total_pixels = cv2.countNonZero(mask_filled)

    # その他の計算（省略）
    W_alpha_original, w_original = lengths[0], lengths[2]
    d_original =2
    theta_2, theta_3 = np.deg2rad(angles[0]), np.deg2rad(angles[1])
    scale_factor = 10 / W_alpha_original

    d_scaled = d_original
    W_alpha_scaled = W_alpha_original * scale_factor
    w_scaled = w_original * scale_factor

    cos_theta_1 = (W_alpha_scaled - w_scaled) / (d_scaled * (np.tan(((math.pi * 1 / 2) - theta_2)) + np.tan(((math.pi * 1 / 2) - theta_3))))
    def sin_from_cos(cos_theta):
        return math.sqrt(1 - cos_theta**2) if cos_theta**2 <= 1 else 0

    sin_theta = sin_from_cos(cos_theta_1)
    
    return points, lengths, angles, sin_theta, total_pixels

def generate_report(quadrilaterals_info, shape_color, img_width, img_height, file_name):
    data = []
    
    img_center_x = img_width / 2
    img_center_y = img_height / 2

    for idx, info in enumerate(quadrilaterals_info):
        points, lengths, angles, sin_theta, total_pixels = info
        theta = np.degrees(math.asin(sin_theta))
        center_y = np.mean(points[:, 1])  # 四角形の中心 y 座標

        position = '+' if points[0][0] > points[2][0] else '-'
        position_y = '+' if center_y > img_center_y else '-'
        if position=='+':
            center_x = np.min(points[:, 0])  # 最も右の x 座標
        if position=='-':
            center_x = np.max(points[:, 0])  # 最も左の x 座標
        
        data.append({
            'File': file_name,
            'Shape': f'{shape_color} {idx + 1}',
            'Theta': theta,
            'CenterX - ImageCenterX': center_x - img_center_x,
            'CenterY - ImageCenterY': center_y - img_center_y,
            'TotalPixels': total_pixels,  # 総ピクセル数を追加
            'Position': position,
            'PositionY': position_y
            
        })
    return pd.DataFrame(data)

def process_directory(input_dir, output_file):
    input_list = list(pathlib.Path(input_dir).glob('**/*.png'))
    if not input_list:
        print("指定したディレクトリに画像が見つかりませんでした。")
        return

    all_data = []
    for img_path in input_list:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"画像を読み込めませんでした: {img_path}")
            continue

        print(f"処理中: {img_path}")
        contours_red, contours_blue, red_mask, blue_mask = detect_shapes(image)
        red_quads = get_quadrilateral_info(contours_red)
        blue_quads = get_quadrilateral_info(contours_blue)

        # 微小範囲の除外
        red_quads_filtered = filter_small_quadrilaterals(red_quads, red_mask, min_area=1500, min_pixels=600)
        blue_quads_filtered = filter_small_quadrilaterals(blue_quads, blue_mask, min_area=1500, min_pixels=600)

        if red_quads_filtered or blue_quads_filtered:
            height, width = image.shape[:2]
            red_quads_info = [calculate_lengths_angles(quad, red_mask) for quad in red_quads_filtered]
            blue_quads_info = [calculate_lengths_angles(quad, blue_mask) for quad in blue_quads_filtered]

            df_red = generate_report(red_quads_info, 'Red', width, height, img_path.name)
            df_blue = generate_report(blue_quads_info, 'Blue', width, height, img_path.name)

            all_data.append(df_red)
            all_data.append(df_blue)
        else:
            print(f"四角形が見つかりませんでした: {img_path}")

    if all_data:
        final_df = pd.concat(all_data).reset_index(drop=True)
        save_to_excel(final_df, output_file)
        print(f"Excelファイルを保存しました: {output_file}")
    else:
        print("処理対象のデータが見つかりませんでした。")



def save_to_excel(df, file_name):
    # Position に基づいてデータを分割
    df_positive = df[df['Position'] == '+']
    df_negative = df[df['Position'] == '-']
    with pd.ExcelWriter(file_name) as writer:
        df_positive.to_excel(writer, sheet_name='Positive', index=False)
        df_negative.to_excel(writer, sheet_name='Negative', index=False)
        

# 実行
input_directory = "c:/studyrep/30degblender/"
output_excel = "30degblender.xlsx"
process_directory(input_directory, output_excel)