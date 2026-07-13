#导入工具包
import numpy as np
import imutils
import cv2
import tkinter as tk
from tkinter import filedialog, simpledialog

# 选择答题卡图片
root = tk.Tk()
root.withdraw()
image_path = filedialog.askopenfilename(
	title="选择答题卡图片",
	filetypes=[("图片文件", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
)
if not image_path:
	print("[ERROR] 未选择图片，程序退出")
	exit()

# 用户输入正确答案
root = tk.Tk()
root.withdraw()
answer_input = tk.simpledialog.askstring("输入正确答案",
	"请输入每题正确答案（用A/B/C/D表示，逗号或空格分隔）\n例如: B,D,A,C",
	parent=root)
if not answer_input:
	print("[ERROR] 未输入答案，程序退出")
	exit()

answer_letters = "ABCD"
answers = [a.strip().upper() for a in answer_input.replace(",", " ").split()]
ANSWER_KEY = {}
for i, a in enumerate(answers):
	if a not in answer_letters:
		print(f"[ERROR] 无效答案 '{a}'，有效选项为 A-D")
		exit()
	ANSWER_KEY[i] = answer_letters.index(a)
num_questions = len(ANSWER_KEY)
num_options = 4

def order_points(pts):
	# 一共4个坐标点
	rect = np.zeros((4, 2), dtype = "float32")

	# 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
	# 计算左上，右下
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# 计算右上和左下
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	return rect

def four_point_transform(image, pts):
	# 获取输入坐标点
	rect = order_points(pts)
	(tl, tr, br, bl) = rect

	# 计算输入的w和h值
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	# 变换后对应坐标位置
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# 计算变换矩阵
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

	# 返回变换后结果
	return warped
def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    if len(cnts) == 0:
        return [], []
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts, boundingBoxes
def get_bubble_fill(thresh_img, contour):
	(x, y, w, h) = cv2.boundingRect(contour)
	cx, cy = x + w // 2, y + h // 2
	radius = int(min(w, h) * 0.45)
	mask = np.zeros(thresh_img.shape, dtype="uint8")
	cv2.circle(mask, (cx, cy), radius, 255, -1)
	filled = cv2.bitwise_and(thresh_img, thresh_img, mask=mask)
	total = cv2.countNonZero(filled)
	circle_area = cv2.countNonZero(mask)
	return total, circle_area
def cv_show(name,img):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()  

# 预处理
image = cv2.imread(image_path)
contours_img = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv_show('blurred',blurred)
edged = cv2.Canny(blurred, 75, 200)
cv_show('edged',edged)

# 轮廓检测
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cv2.drawContours(contours_img,cnts,-1,(0,0,255),3) 
cv_show('contours_img',contours_img)
docCnt = None

# 确保检测到了
if len(cnts) > 0:
	# 根据轮廓大小进行排序
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

	# 遍历每一个轮廓
	for c in cnts:
		# 近似
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)

		# 准备做透视变换
		if len(approx) == 4:
			docCnt = approx
			break

# 执行透视变换

warped = four_point_transform(gray, docCnt.reshape(4, 2))
cv_show('warped',warped)
# Otsu's 阈值处理
thresh = cv2.threshold(warped, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] 
cv_show('thresh',thresh)
thresh_Contours = thresh.copy()

# 形态学闭运算：补全气泡圆环的断裂，使轮廓更完整
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
thresh_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
cv_show('thresh_closed', thresh_closed)

# 使用 RETR_TREE 层级检测，获取完整轮廓层级关系
cnts, hierarchy = cv2.findContours(thresh_closed.copy(), cv2.RETR_TREE,
	cv2.CHAIN_APPROX_SIMPLE)

# 可视化所有轮廓
all_contours_img = thresh_closed.copy()
all_contours_img = cv2.cvtColor(all_contours_img, cv2.COLOR_GRAY2BGR)
for i, c in enumerate(cnts):
	color = (0, 255, 0) if hierarchy[0][i][3] == -1 else (255, 0, 0)
	cv2.drawContours(all_contours_img, [c], -1, color, 1)
cv_show('all_contours', all_contours_img)

# 筛选气泡候选：收集所有近圆形轮廓的外接框
candidates = []
for i, c in enumerate(cnts):
	(x, y, w, h) = cv2.boundingRect(c)
	if w < 10 or h < 10:
		continue
	ar = w / float(h)
	if ar < 0.7 or ar > 1.4:
		continue
	area = cv2.contourArea(c)
	candidates.append((x, y, w, h, area, c, i))

print(f"[DEBUG] 候选轮廓数: {len(candidates)}")
for i, (x, y, w, h, area, c, idx) in enumerate(candidates[:30]):
	print(f"  候选{i}: x={x}, y={y}, w={w}, h={h}, area={area:.0f}, ar={w/h:.2f}")

# 动态尺寸筛选：取尺寸中位数，保留中位数附近的轮廓
if len(candidates) > 0:
	ws = [w for x, y, w, h, area, c, idx in candidates]
	median_w = np.median(ws)
	min_w = median_w * 0.6
	max_w = median_w * 1.5
	candidates = [cand for cand in candidates if min_w <= cand[2] <= max_w and min_w <= cand[3] <= max_w]
	print(f"[DEBUG] 尺寸筛选后候选数: {len(candidates)}, median_w={median_w:.0f}, range=[{min_w:.0f}, {max_w:.0f}]")

# 去重：同一位置的多个轮廓（外圆环+内字母）只保留最大的那个
candidates.sort(key=lambda c: c[4], reverse=True)  # 按面积从大到小
questionCnts = []
used_centers = []
for x, y, w, h, area, c, idx in candidates:
	cx, cy = x + w//2, y + h//2
	# 检查是否与已选气泡中心太近
	too_close = False
	for ux, uy in used_centers:
		dist = np.sqrt((cx - ux)**2 + (cy - uy)**2)
		if dist < min(w, h) * 0.4:
			too_close = True
			break
	if not too_close:
		questionCnts.append(c)
		used_centers.append((cx, cy))

print(f"[DEBUG] 去重后气泡数: {len(questionCnts)}")

# 绘制检测到的气泡
bubble_debug = warped.copy()
if len(bubble_debug.shape) == 2:
	bubble_debug = cv2.cvtColor(bubble_debug, cv2.COLOR_GRAY2BGR)
for c in questionCnts:
	(x, y, w, h) = cv2.boundingRect(c)
	cv2.rectangle(bubble_debug, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv_show('detected_bubbles', bubble_debug)

# 多列布局：按x坐标分列，每列内按y排序分组
def cluster_into_columns(bubbles, num_options=4, num_questions=None):
	if len(bubbles) == 0:
		return []

	# 按x坐标排序，找列间距
	sorted_by_x = sorted(bubbles, key=lambda c: cv2.boundingRect(c)[0])
	xs = [cv2.boundingRect(c)[0] for c in sorted_by_x]

	# 计算同一行内相邻选项的x间距（列内间距）
	# 同一题的选项间距 < 不同列的间距
	# 用所有相邻x差的中位数的1.5倍作为列内最大间距
	diffs = [xs[i+1] - xs[i] for i in range(len(xs)-1)]
	diffs.sort()
	median_diff = diffs[len(diffs)//2] if diffs else 0

	# 估算平均气泡宽度
	widths = [cv2.boundingRect(c)[2] for c in bubbles]
	avg_w = sum(widths) / len(widths) if widths else 20

	# 列内选项最大间距 = 选项中心间距 ~ 2~3倍气泡宽
	# 列间距明显更大
	col_gap_threshold = avg_w * 3

	print(f"[DEBUG] 平均气泡宽度: {avg_w:.0f}, 列间距阈值: {col_gap_threshold:.0f}")

	# 按x间距分组：间距 > col_gap_threshold 就分到下一列
	columns = [[sorted_by_x[0]]]
	for i in range(1, len(sorted_by_x)):
		gap = xs[i] - xs[i-1]
		if gap > col_gap_threshold:
			columns.append([])
		columns[-1].append(sorted_by_x[i])

	num_cols = len(columns)
	print(f"[DEBUG] 检测到 {num_cols} 列，每列气泡数: {[len(col) for col in columns]}")

	# 每列内按y排序
	all_questions = []
	for col_idx, col in enumerate(columns):
		# 列内按y坐标从上到下排序
		col_sorted = sort_contours(col, method="top-to-bottom")[0]
		# 每 num_options 个一组 = 一道题
		for j in range(0, len(col_sorted), num_options):
			group = col_sorted[j:j+num_options]
			if len(group) == num_options:
				# 组内按x从左到右排序 = A/B/C/D
				group_sorted = sort_contours(group, method="left-to-right")[0]
				all_questions.append(group_sorted)

	print(f"[DEBUG] 组装完成，共 {len(all_questions)} 题")

	# 可视化列分配
	col_debug = warped.copy()
	if len(col_debug.shape) == 2:
		col_debug = cv2.cvtColor(col_debug, cv2.COLOR_GRAY2BGR)
	colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
	for col_idx, col in enumerate(columns):
		color = colors[col_idx % len(colors)]
		for c in col:
			(x, y, w, h) = cv2.boundingRect(c)
			cv2.rectangle(col_debug, (x, y), (x+w, y+h), color, 2)
		cv2.putText(col_debug, f"Col{col_idx+1}", (cv2.boundingRect(col[0])[0], 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
	cv_show('columns_debug', col_debug)

	return all_questions

all_questions = cluster_into_columns(questionCnts, num_options, num_questions)
warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
detected_questions = len(all_questions)
if detected_questions != num_questions:
	print(f"[WARNING] 检测到 {detected_questions} 题，但输入了 {num_questions} 个答案")
correct = 0

for q in range(min(len(all_questions), num_questions)):
	cnts = all_questions[q]
	bubbled = None

	# 遍历每一个结果
	for (j, c) in enumerate(cnts):
		total, _ = get_bubble_fill(thresh, c)
		if bubbled is None or total > bubbled[0]:
			bubbled = (total, j)

	# 判断是否填涂：最高填涂量需明显高于其他气泡
	totals = []
	for c in cnts:
		total, _ = get_bubble_fill(thresh, c)
		totals.append(total)
	max_total = max(totals)
	others = sum(totals) - max_total
	avg_others = others / (len(totals) - 1) if len(totals) > 1 else 0
	# 填涂判定：最高值是其他平均值的1.3倍以上
	is_bubbled = max_total > avg_others * 1.3 and max_total > avg_others + 50
	print(f"[DEBUG] Q{q+1}: totals={totals}, max={max_total}, avg_others={avg_others:.0f}, filled={is_bubbled}")

	# 对比正确答案
	color = (0, 0, 255)
	k = ANSWER_KEY.get(q, 0)
	if q not in ANSWER_KEY:
		continue

	# 绘制学生填涂的选项（蓝色），未填涂则不绘制
	if is_bubbled:
		cv2.drawContours(warped, [cnts[bubbled[1]]], -1, (255, 0, 0), 2)

	# 判断正确（未填涂视为错误）
	if is_bubbled and k == bubbled[1]:
		color = (0, 255, 0)
		correct += 1

	# 绘制正确答案（正确绿色，错误红色）
	cv2.drawContours(warped, [cnts[k]], -1, color, 3)

# 输出学生答案
student_answers = []
for q in range(min(len(all_questions), num_questions)):
	cnts = all_questions[q]
	totals = []
	for c in cnts:
		total, _ = get_bubble_fill(thresh, c)
		totals.append(total)
	max_total = max(totals)
	others = sum(totals) - max_total
	avg_others = others / (len(totals) - 1) if len(totals) > 1 else 0
	is_bubbled = max_total > avg_others * 1.3 and max_total > avg_others + 50
	if is_bubbled:
		student_answers.append(answer_letters[totals.index(max_total)])
	else:
		student_answers.append("-")

print("[INFO] 学生答案:", " ".join(student_answers))
print("[INFO] 正确答案:", " ".join([answer_letters[ANSWER_KEY[q]] for q in range(len(ANSWER_KEY))]))
score = (correct / num_questions) * 100
print("[INFO] score: {:.2f}%".format(score))
cv2.putText(warped, "{:.2f}%".format(score), (10, 30),
	cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow("Original", image)
cv2.imshow("Exam", warped)
cv2.waitKey(0)