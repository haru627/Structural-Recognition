import cv2
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


def detect_shapes(image):
    blurred = cv2.GaussianBlur(image, (1,1), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 赤領域のマスク
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([60, 255, 255])
    lower_red2 = np.array([140, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.add(red_mask1, red_mask2)

    # 青領域のマスク
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # GaussianBlurでノイズ除去
    red_mask = cv2.GaussianBlur(red_mask, (1, 1), 0)
    blue_mask = cv2.GaussianBlur(blue_mask, (1, 1), 0)

    # 輪郭の検出
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours_red, contours_blue

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

def calculate_lengths_angles(quadrilateral):
    points = quadrilateral.reshape(4, 2)
    lengths = [np.linalg.norm(points[i] - points[(i + 1) % 4]) for i in range(4)]

    # 最長の辺を探し、その辺が AB になるように頂点を並び替え
    max_length_idx = np.argmax(lengths)
    points = np.roll(points, -max_length_idx, axis=0)

    lengths, angles = [], []
    for i in range(4):
        pt1, pt2 = points[i], points[(i + 1) % 4]
        lengths.append(np.linalg.norm(pt1 - pt2))

    for i in range(4):
        # 頂点順序を基準に角度を計算
        pt_prev = points[i - 1]  # 頂点iの1つ前 (反時計回りの隣接点)
        pt_current = points[i]  # 現在の頂点
        pt_next = points[(i + 1) % 4]  # 頂点iの次 (反時計回りの隣接点)

        # ベクトル計算
        v1 = pt_prev - pt_current
        v2 = pt_next - pt_current

        # 内積と角度計算
        angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
        angles.append(np.degrees(angle))

    W_alpha_original, w_original = lengths[0], lengths[2]
    d_original = 1
    theta_2, theta_3 = np.deg2rad(angles[0]), np.deg2rad(angles[1])
    scale_factor = 5 / W_alpha_original
 
    d_scaled = d_original
    W_alpha_scaled = W_alpha_original * scale_factor
    w_scaled = w_original * scale_factor

    cos_theta_1 = (W_alpha_scaled - w_scaled) / (d_scaled * (np.tan(((math.pi * 1 / 2) - theta_2)) + np.tan(((math.pi * 1 / 2) - theta_3))))
    if np.tan(((math.pi*1/2)-theta_2)) +  np.tan(((math.pi*1/2)-theta_3))<0.001:
            cos_theta_1=1
    def sin_from_cos(cos_theta):
        return math.sqrt(1 - cos_theta ** 2) if cos_theta ** 2 <= 1 else 0

    sin_theta = sin_from_cos(cos_theta_1)
    return points, lengths, angles, sin_theta

def draw_shape_info(image, points, shape_name, color, index):
    leftmost_x = points[np.argmin(points[:, 0]), 0]
    print(color)
    # y 座標は中心 (平均値) を計算
    center_y = int(np.mean(points[:, 1]))
    if color==(0, 0, 255):
        center_y=center_y+100
    if color==(255, 0, 0):
        center_y=center_y-100
    # 描画位置を設定 (x は左端、y は中心)
    text_position = (int(leftmost_x), center_y)
    black=(255,255,255)
    cv2.putText(image, f'{shape_name} {index + 1}', tuple(text_position), cv2.FONT_HERSHEY_TRIPLEX,1.5, black, 3)
    return image

def compare_shapes(quadrilaterals_info_red, quadrilaterals_info_blue, image):
    for idx, red_info in enumerate(quadrilaterals_info_red):
        image = draw_shape_info(image, red_info[0], 'Red', (0, 0, 255), idx)
    for idx, blue_info in enumerate(quadrilaterals_info_blue):
        image = draw_shape_info(image, blue_info[0], 'Blue', (255, 0, 0), idx)
    return image

def set_window_size_to_image(window_name, image, scale=1):
    resized_image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
    cv2.imshow(window_name, resized_image)

def generate_report(quadrilaterals_info, shape_color, image_center):
    data = []
    for idx, info in enumerate(quadrilaterals_info):
        points, lengths, angles, sin_theta = info
        theta = np.degrees(math.asin(sin_theta))
        
        # 四角形の中心座標
        center_x = np.mean(points[:, 0])

        # 画像中心座標
        image_center_x = image_center[0]
        position = '+' if points[0][0] > points[2][0] else '-'
        # centerX - ImageCenterX の差を計算
        center_x_diff = center_x - image_center_x
        
        # Positionパラメータをそのまま残しつつ、新しいパラメータcenter_x_diffを追加
        data.append({
            'Shape': f'{shape_color} {idx + 1}',
            'Length_AB': lengths[0],
            'Angle_A': angles[0],
            'Angle_B': angles[1],
            'Angle_C': angles[2],
            'Angle_D': angles[3],
            'sin_theta': sin_theta,
            'theta': theta,
            'Position': position,  # 元のPositionを保持
            'CenterX_diff': center_x_diff  # 新しいパラメータ
        })
    df = pd.DataFrame(data)
    print(f"\n{shape_color} Quadrilateral Report:")
    print(df)
    return df

def generate_final_report(df_red, df_blue):
    # 画像中心に基づいてソート
    final_df = pd.concat([df_red, df_blue]).sort_values(by=['CenterX_diff'], ascending=True).reset_index(drop=True)

    print("\nFinal Combined Report:")
    print(final_df)
    return final_df

def plot_points_from_report(df, closest_angles, d_original):
    points = [(0, 0)]
    
    for angle, (_, row) in zip(closest_angles, df.iterrows()):
        x_prev, y_prev = points[-1]
        theta = np.radians(angle)  # 最も近い角度を使用
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        if row['Position'] == '-':
            sin_theta = -sin_theta
        x_new = x_prev + d_original * cos_theta
        y_new = y_prev + d_original * sin_theta
        points.append((x_new, y_new))
    
    x_coords, y_coords = zip(*points)
    plt.figure(figsize=(8, 6))
    plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='b', markersize=8, linewidth=2)
    for i, (x, y) in enumerate(points):
        plt.text(x, y, f'({x:.2f}, {y:.2f})', fontsize=10, ha='right')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Line Graph of Points with Adjusted Angles')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def find_two_closest_quadratics(theta, center_x_diff, quadratic_coefficients, angles, position):
    distances = []
    
    # Positionが'+'の場合、center_x_diffの符号を反転
    if '+' in position:
        center_x_diff = -center_x_diff

    for i, (a, b, c) in enumerate(quadratic_coefficients):
        predicted_y = a * center_x_diff**2 + b * center_x_diff + c
        distance = abs(theta - predicted_y)
        distances.append((distance, i))
    
    # 距離でソートして最も近い二つを取得
    distances.sort(key=lambda x: x[0])
    closest_two = distances[:2]
    
    return closest_two

def estimate_angle_from_closest_quadratics(theta, center_x_diff, closest_two, angles):
    # 最も近い二つのインデックスと距離を取得
    (dist1, idx1), (dist2, idx2) = closest_two
    angle1 = angles[idx1]
    angle2 = angles[idx2]

    # 距離の比を使用してθを推定する
    ratio = dist1 / dist2 if dist2 != 0 else 1  # 距離比
    estimated_angle = (ratio * angle1 + angle2) / (ratio + 1)
    
    return estimated_angle


def main(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    print(f"画像のx軸方向の総ピクセル数: {width}")
    
    image_center = np.array([width / 2, height / 2])
    
    contours_red, contours_blue = detect_shapes(image)
    red_quads = get_quadrilateral_info(contours_red)
    blue_quads = get_quadrilateral_info(contours_blue)

    if red_quads or blue_quads:
        red_quads_info = [calculate_lengths_angles(quad) for quad in red_quads]
        blue_quads_info = [calculate_lengths_angles(quad) for quad in blue_quads]

        output_image = compare_shapes(red_quads_info, blue_quads_info, image)

        window_name = 'Image with Shape Info'
        set_window_size_to_image(window_name, output_image, scale=0.25)

        df_red = generate_report(red_quads_info, 'Red', image_center)
        df_blue = generate_report(blue_quads_info, 'Blue', image_center)

        final_df = generate_final_report(df_red, df_blue)

        quadratic_coefficients = [
        (-0.0000012229, 0.0071141389, 6.229),
        (-0.0000016278, 0.0090356386, 11.821),
        (-0.0000018355, 0.0105389306, 15.279),
        (-0.0000014764, 0.0101280169, 21.170),
        (-0.0000019893, 0.0119350472, 25.243),
        (-0.0000020655, 0.0119056001, 30.696),
        (-0.0000015674, 0.0110666738, 35.864),
        (-0.0000011517, 0.0106282789, 40.815),
        (-0.0000009426, 0.0104888732, 45.706),
        (-0.0000007950, 0.0104285865, 50.558),
        (-0.0000006468, 0.0104256763, 55.335),
        (-0.0000005294, 0.0102962317, 60.452),
        (-0.0000005292, 0.0103061363, 65.368),
        (-0.0000004457, 0.0102143733, 70.308),
        (-0.0000004727, 0.0100276028, 75.316),
        (-0.0000003913, 0.0098646508, 80.160),
        (-0.0000005935, 0.0091321686, 84.708)
        ]


        theta_values = final_df['theta'].tolist()
        center_x_diff_values = final_df['CenterX_diff'].tolist()
        position_values = final_df['Position'].tolist()

        estimated_angles = []
        angles = [ 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,65,70,75,80,85]

        for theta, center_x_diff, position in zip(theta_values, center_x_diff_values, position_values):
            closest_two = find_two_closest_quadratics(theta, center_x_diff, quadratic_coefficients, angles, position)
            estimated_angle = estimate_angle_from_closest_quadratics(theta, center_x_diff, closest_two, angles)
            estimated_angles.append(estimated_angle)
            print(f"theta: {theta}, CenterX_diff: {center_x_diff} → 推定角度: {estimated_angle:.2f}°")

        plot_points_from_report(final_df, estimated_angles, d_original=2)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

# image_pathは画像のファイルパスに置き換えてください
image_path = 'testd.png'
main(image_path)
