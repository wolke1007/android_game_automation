import base64
import functools
import json
import os
import shutil
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
import pytesseract
import cv2
from appium.webdriver.common.mobileby import MobileBy
from appium.webdriver.common.touch_action import TouchAction
from beartype import beartype
from selenium.common import TimeoutException, NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait

from int_forestage_module.import_ml_core import import_ext_module
from concurrent.futures import ThreadPoolExecutor
import uiautomator2 as u2
from appium import webdriver as appium_webdriver

DEBUG = True
HEADLESS = True
RAISE_E = True
TARGET_PICS_DIR = './target_pics/'
fail_cases = []
PC_CLOSE_WINDOW_TYPE_1_OFFSET = (150, 0)
PC_CLOSE_WINDOW_TYPE_2_OFFSET = (170, 0)
MB_CLOSE_WINDOW_TYPE_1_OFFSET = (40, 0)
MB_CLOSE_WINDOW_TYPE_2_OFFSET = (40, 0)
PC_FIRST_BET_RECORD_OFFSET = (0, 150)
MB_FIRST_BET_RECORD_OFFSET = (0, 50)


def printlog(msg: str, is_debug: bool = False):
    if is_debug:
        print(msg)


def image_recognition(img_gray, template_path, template_threshold):
    '''
    此為找尋圖片 x y 位置的演算法
    https://www.cnblogs.com/bruce1992/p/16485725.html
    screenshot_path:待检测的图片
    template_path:模板小图
    template_threshold:模板匹配的置信度
    '''
    # 标注位置
    img_disp = img_gray.copy()
    template_img = cv2.imread(template_path, 0)  # 模板小图
    if img_disp is None or template_img is None:
        raise RuntimeError(
            f"img_disp or template_img is None, please check img_disp: {img_disp}, template_img: {template_img}")
    h, w = template_img.shape[:2]
    res = cv2.matchTemplate(img_gray, template_img, cv2.TM_CCOEFF_NORMED)
    # 筛选大于一定匹配值的点
    val, result = cv2.threshold(res, template_threshold, 1.0, cv2.THRESH_BINARY)
    match_locs = cv2.findNonZero(result)
    if match_locs is not None:
        a = []
        b = []
        for match in match_locs:
            match_loc = match[0]
            # 注意计算右下角坐标时x坐标要加模板图像shape[1]表示的宽度，y坐标加高度
            right_bottom = (match_loc[0] + template_img.shape[1], match_loc[1] + template_img.shape[0])
            x = int((match_loc[0] + right_bottom[0]) / 2)
            y = int((match_loc[1] + right_bottom[1]) / 2)
            a.append((x, y))
            b.append(res[match_loc[1], match_loc[0]])
            # ===========debug=======
            cv2.rectangle(img_disp, match_loc, right_bottom, (0, 255, 0), 5, 8, 0)
            cv2.circle(result, match_loc, 10, (255, 0, 0), 3)
            # ===========debug=======
        data = [(loc, threshold) for loc, threshold in zip(a, b)]
        # max_array, max_threshold = max(l, key=lambda i: i[1])
        # print(a)  # debug
        # print(b)  # debug
        # =================debug============
        # print(max_array)  # debug
        # print(max_threshold)  # debug
        # cv2.imshow('Matching Result', img_disp)  # debug
        # cv2.waitKey(0)  # debug
        # cv2.destroyAllWindows()  # debug
        # ==================

        # 排除 x, y 同時相差小於 10 的數據，若要選擇時選擇批配度高的
        # 例如
        # ((687, 1009), 0.9022711), ((686, 1007), 0.88157743)
        # 選擇 ((687, 1009), 0.9022711) 保留

        filtered_data = []
        processed_indices = set()

        for i, (coord1, score1) in enumerate(data):
            if i in processed_indices:
                continue
            best_score = score1
            best_index = i

            for j, (coord2, score2) in enumerate(data):
                if i != j and j not in processed_indices:
                    x1, y1 = coord1
                    x2, y2 = coord2
                    x_diff = abs(x1 - x2)
                    y_diff = abs(y1 - y2)

                    if x_diff < 10 and y_diff < 10:
                        if score2 > best_score:
                            best_score = score2
                            best_index = j
                        processed_indices.add(j)

            filtered_data.append(data[best_index])

        print(filtered_data)  # debug

        return filtered_data, img_disp
    else:
        return None, None


def find_pic_x_y(screenshot, using_base64: bool, template_path, template_threshold):
    """
    目標是找到相似度 高於 並 最接近 template_threshold 的一個物件
    """
    if using_base64:
        img_data = base64.b64decode(screenshot)
        buffer_np = np.frombuffer(img_data, dtype=np.uint8)
        img_rgb = cv2.imdecode(buffer_np, cv2.IMREAD_COLOR)
        img_gray = cv2.imdecode(buffer_np, cv2.IMREAD_GRAYSCALE)
    else:
        img_rgb = cv2.imread(screenshot)  # 需要检测的图片
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)  # 转化成灰色

    match_set, img = image_recognition(img_gray, template_path, template_threshold)
    if match_set is not None:
        match_array, match_threshold = max(match_set, key=lambda i: i[1])
        template_name = os.path.split(template_path)[-1]
        print(f"準確度 {match_threshold} 找到{template_name}物件, {match_array}")  # debug
        x, y = match_array
        return int(x), int(y), img
    else:
        return None


def find_pics_x_y(screenshot, using_base64: bool, template_path, template_threshold):
    """
    目標是找到相似度 高於 template_threshold 的 每個 物件
        e.g. [(x1,y1), (x2,y2)...]
    """
    if using_base64:
        img_data = base64.b64decode(screenshot)
        buffer_np = np.frombuffer(img_data, dtype=np.uint8)
        img_rgb = cv2.imdecode(buffer_np, cv2.IMREAD_COLOR)
        img_gray = cv2.imdecode(buffer_np, cv2.IMREAD_GRAYSCALE)
    else:
        img_rgb = cv2.imread(screenshot)  # 需要检测的图片
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)  # 转化成灰色

    match_set, img = image_recognition(img_gray, template_path, template_threshold)
    template_name = os.path.split(template_path)[-1]
    if match_set is not None:
        print(f"準確度 {template_threshold} 找到{len(match_set)}個物件, {match_set}")  # debug
        return list(match_set), img
    else:
        return None


def method_setup_teardown(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 使用 debug 時 breadcrumbs 會從 args 的第二個參數傳進來，第一個則是 GameH5 的實體
        print('args: ', args)
        test_class_instance = args[0]
        failed_pic = os.path.join(test_class_instance.screenshot_root_save_path, f'failed_debug.png')

        # 刪除上次的 debug pic
        if os.path.exists(failed_pic):
            print(f"刪除上次的 debug pic，{failed_pic}")
            os.remove(failed_pic)

        try:
            test_class_instance.setup_method(func)  # 自己 handle pytest 的 setup
            start_time = time.time()
            func(*args, **kwargs)
            shutil.rmtree(test_class_instance.debug_dir)  # 如果成功就刪除debug資料夾
            printlog(f"{func.__name__} time cost: {time.time() - start_time}", is_debug=DEBUG)
        except Exception as e:
            printlog(f'======================={traceback.format_exc()}=====================', is_debug=DEBUG)
            # 失敗後截圖
            test_class_instance.page_driver.take_screenshot(failed_pic)
            # obj_name = 'mb_game' if 'mb_' in func.__name__ else 'pc_game'
            # rerun_command = f'{obj_name}.{func.__name__}("{account}", "{password}", "{domain}", "{language}")'
            # test_class_instance.fail_cases_queue.put((e.__repr__(), rerun_command))
            if RAISE_E:
                raise e  # debug
        finally:
            test_class_instance.teardown_method(func)  # 自己 handle pytest 的 teardown

    return wrapper


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class AppiumWebdriverAndroid(metaclass=Singleton):

    def __init__(self, data):
        self.data = data
        self.driver = None
        SDK_ENV = data.get('SDK_ENV', 'Appium')
        ANDROID_EMULATOR_ID = data['ANDROID_EMULATOR_ID']
        ANDROID_PACKAGE_NAME = data.get('ANDROID_PACKAGE_NAME', 'com.dafacloud.meeline')
        ANDROID_ACTIVITY_NAME = data.get('ANDROID_ACTIVITY_NAME', "mvp.ui.activity.SplashActivity")
        TIMEOUT = data.get('TIMEOUT', 2)
        APPIUM_PORT = data.get('APPIUM_PORT', 4876)
        NO_RESET = data.get('NO_RESET', False)

        path = self.data.get('path')
        self.appium_process = None

        if SDK_ENV == 'Appium':
            desired_capabilities = {
                "appActivity": ANDROID_ACTIVITY_NAME,
                "appPackage": ANDROID_PACKAGE_NAME,
                "platformName": "Android",
                "udid": ANDROID_EMULATOR_ID,
                "newCommandTimeout": 600,
                "automationName": "UIAutomator2",
                "systemPort": 8201,
                "noReset": NO_RESET,

            }
            print("desired_capabilities: ", desired_capabilities)
            time.sleep(3)  # 等待 appium server 起好
            self.driver = appium_webdriver.Remote(command_executor=f'http://localhost:{APPIUM_PORT}',
                                                  desired_capabilities=desired_capabilities)
            self.driver.implicitly_wait(TIMEOUT)
        else:
            raise EnvironmentError(f"ANDROID_ENV 環境變數沒有設定正確，當前為: {SDK_ENV}")


class SeleniumDriver(metaclass=Singleton):

    def __init__(self, selenium_driver):
        self.driver = selenium_driver  # appium or selenium 的原生 webdriver

    @beartype
    def _wait_until_element_presence(self, element: tuple, wait_time=1):
        if element[0][0] == By.XPATH or element[0][0] == By.ID:
            wait = WebDriverWait(self.driver, wait_time)
            try:
                # 網路上找到等待元素出現的 best practice
                return wait.until(
                    expected_conditions.presence_of_element_located(element[0])
                )
            except TimeoutException as e:
                raise LookupError(f"於頁面上經過 {wait_time} 秒找不到 {element[1]} ")
        # 下半部為為了支援其他 ios 獨有的 selector 實作的
        elif element[0][0] == MobileBy.IOS_CLASS_CHAIN:
            web_element = self.driver.find_elements_by_ios_class_chain(element[0][1])
            if len(web_element) != 1:
                raise ValueError(f"{element[0][1]}找到{str(len(web_element))}個元素")
            return web_element[0]
        elif element[0][0] == MobileBy.IOS_PREDICATE:
            web_element = self.driver.find_elements_by_ios_predicate(element[0][1])
            if len(web_element) != 1:
                raise ValueError(f"{element[0][1]}找到{str(len(web_element))}個元素")
            return web_element[0]
        else:
            raise NotImplementedError(f"{element[0][0]} 此種 selector 還未實現")

    @beartype
    def click(self, element: tuple, wait_time=10, delay=1, retry=1):
        """
        wait_time 等待出現的時間
        delay 睡多久後才執行
        retry 失敗後重試次數
        """
        if not retry:
            raise LookupError(f"頁面上沒找到元素: {element}")
        time.sleep(delay)
        try:
            self._wait_until_element_presence(element, wait_time).click()
            printlog(f"點擊 [{element[1]}]")  # debug
        except Exception:
            retry -= 1
            if retry:
                printlog(f"retry: {retry}")  # debug
            self.click(element=element, wait_time=wait_time, retry=retry)

    @beartype
    def click_if_exist(self, element: tuple, wait_time=1, retry=1) -> bool:
        """
        若存在則點擊，不存在也不會報錯的情境使用
        像是公告頁面的關閉按鈕
        或是第一次開啟微聊要求資料夾權限的按鈕
        """
        try:
            self.click(element=element, wait_time=wait_time, retry=retry)
            return True
        except LookupError:
            printlog(f"頁面上沒找到元素: {element}, 跳過此步驟不點擊")
            return False

    @beartype
    def sendkeys(self, text, element, wait_time=10, delay=1):
        time.sleep(delay)
        self._wait_until_element_presence(element, wait_time).clear()
        self._wait_until_element_presence(element, wait_time).send_keys(text)
        printlog(f"於 [{element[1]}] 輸入字串： [{text}]")  # debug

    @beartype
    def is_element_exist(self, element: tuple, wait_time=10, retry=1):
        if not retry:
            return False
        try:
            self._wait_until_element_presence(element, wait_time)
            printlog(f"頁面上可以找到: {element[1]}")
            return True
        except (LookupError, ValueError):
            retry -= 1
            if retry:
                printlog(f"retry: {retry}")  # debug
            return self.is_element_exist(element, wait_time, retry)

    @beartype
    def find_elements_text_list(self, element: tuple) -> list:
        """
        使用 xpath 搜尋頁面上的元素，並將符合的元素字串組成list回傳
        例. ['2022年02月07日 15:32', '2022年02月07日 17:52', '2022年02月07日 18:52']
        """
        if element[0][0] != 'xpath':
            raise KeyError(f"一定得使用 xpath 來搜尋: {element[0][0]}")
        elements = []
        index = 1
        while True:
            element_xpath = f"({element[0][1]})[{index}]"
            try:
                found_element = self.driver.find_element('xpath', value=element_xpath)
                elements.append(found_element)
                index += 1
            except NoSuchElementException:
                break
        if elements:
            element_text = [e.text for e in elements] if len(elements) > 0 else []
            return element_text
        else:
            return []

    @beartype
    def swipe(self, start_x, start_y, end_x, end_y, duration=300):
        """直接在頁面上做 x1 y1 -> x2 y2 點對點的滑動"""
        printlog(f"滑動頁面")
        self.driver.swipe(start_x, start_y, end_x, end_y, duration)

    @beartype
    def element_text_show_after_swipe(self, element: tuple, text_collect: set, delay_time=5, loop=1) -> set:
        """
        使用情境為在對話歷史紀錄上滑動畫面找元素
        """
        printlog(f"準備進行滑動頁面 {loop} 次，間隔 {delay_time} 秒")
        for i in range(loop):
            text_collect = text_collect.union(
                set(self.find_elements_text_list(element)))
            self.swipe(start_x=350, start_y=300, end_x=350, end_y=980, duration=500)
            time.sleep(delay_time)
            text_collect = text_collect.union(
                set(self.find_elements_text_list(element)))
        return text_collect

    @beartype
    def tap_screen(self, x, y, duration=50):
        """
        直接點擊螢幕
        不建議直接這樣做，除非頁面上找不到該元素進行操作
        """
        TouchAction(self.driver).tap(None, x, y, 1).perform()

    @beartype
    def swipe_to_up(self, element: tuple, distance=100):
        """點著指定元素向上滑"""
        element_x_y = self._wait_until_element_presence(element).location
        self.swipe(element_x_y.get('x'), element_x_y.get('y'),
                   element_x_y.get('x'), element_x_y.get('y') - distance)

    @beartype
    def swipe_to_down(self, element: tuple, distance=100):
        """點著指定元素向下滑"""
        element_x_y = self._wait_until_element_presence(element).location
        self.swipe(element_x_y.get('x'), element_x_y.get('y'),
                   element_x_y.get('x'), element_x_y.get('y') + distance)

    @beartype
    def swipe_to_left(self, element: tuple, distance=100):
        """點著指定元素向左滑"""
        element_x_y = self._wait_until_element_presence(element).location
        self.swipe(element_x_y.get('x'), element_x_y.get('y'),
                   element_x_y.get('x') - distance, element_x_y.get('y'))

    @beartype
    def swipe_to_right(self, element: tuple, distance=100):
        """點著指定元素向右滑"""
        element_x_y = self._wait_until_element_presence(element).location
        self.swipe(element_x_y.get('x'), element_x_y.get('y'),
                   element_x_y.get('x') + distance, element_x_y.get('y'))

    @beartype
    def get_location(self, element, wait_time) -> dict:
        """
        return {'x': 1, 'y': 1}
        """
        return self._wait_until_element_presence(element, wait_time).location

    @beartype
    def press_keycode(self, code: int):
        return self.driver.press_keycode(code)

    @beartype
    def take_screenshot(self, path):
        return self.driver.save_screenshot(path)

    @beartype
    def click_x_y(self, x: int, y: int, offset: tuple = (0, 0)):
        # 创建 TouchAction 对象
        action = TouchAction(self.driver)
        offset_x, offset_y = offset
        new_x = x + offset_x
        new_y = y + offset_y
        # 在指定坐标进行点击
        action.tap(x=new_x, y=new_y).perform()

    def screenshot_to_base64(self):
        screenshot_base64 = self.driver.get_screenshot_as_base64()
        # 解码Base64字符串为字节数据
        screenshot_bytes = base64.b64decode(screenshot_base64)
        # 将字节数据转换为NumPy数组
        buffer_np = np.frombuffer(screenshot_bytes, dtype=np.uint8)
        # 使用OpenCV的cv2.imdecode函数解码为图片
        img = cv2.imdecode(buffer_np, cv2.IMREAD_COLOR)

        return img, screenshot_base64


class AndroidGame:

    def __init__(self):
        self.start_time = time.time()
        self.appium_process = None
        print("------ setup before class ------")

        self.kill_appium_process()
        self.kill_adb_process()
        self.kill_phone_process()

        # 在子線程啟動 Appium Server
        executor = ThreadPoolExecutor(max_workers=2)
        future = executor.submit(self.start_appium_server)
        now = datetime.now().strftime("%H:%M:%S")
        print(f'等待 Appium Server 啟動完成，否則腳本會因為連不到 server 而噴錯: {now}')
        time.sleep(15)  # 等待 Appium Server 啟動完成，否則腳本會因為連不到 server 而噴錯

        selenium_driver = AppiumWebdriverAndroid({
            'ANDROID_EMULATOR_ID': 'HT7B71A02138',
            'ANDROID_PACKAGE_NAME': 'com.hortor.fejsf.asia',
            'ANDROID_ACTIVITY_NAME': "org.cocos2dx.javascript.SplashActivity",
            'NO_RESET': True,
        }).driver
        self.page_driver = SeleniumDriver(selenium_driver)
        self.screenshot_save_path = None
        self.screenshot_root_save_path = os.path.join(os.getcwd(), 'screenshots')
        self.template_image_root_path = None
        self.debug_dir = None
        self.http_response = {}
        self.appium_process = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        printlog(f"total test time cost: {time.time() - self.start_time}", is_debug=DEBUG)
        # 刪除
        self.kill_appium_process()
        self.kill_adb_process()
        self.kill_phone_process()

    def start_appium_server(self):
        command = "appium --port 4876 > /dev/null 2>&1 &"
        print(command)
        self.appium_process = subprocess.Popen(command, shell=True, stdout=None)

    def kill_phone_process(self):
        # 刪除手機中的 atx-agent 與 apkagent.cli
        #   atx-agent server --nouia -d
        #   apkagent.cli
        # 這兩個要是在運行 uiautomator2 後沒有刪除的話會造成 Appium 的 uiautomator2 運行失敗
        command = "adb shell 'pkill -f agent'"
        subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(command)
        time.sleep(1)  # 不停頓確定有機會造成 uiautomator2 執行錯誤

    def kill_appium_process(self):
        # 刪除 Appium 與 adb 留下的程序，不然執行 Appium 的 uiautomator2 會出錯
        # 程序長得像是這樣:
        #   /Users/cloudchen/Library/Android/sdk/platform-tools/adb -P 5037 -s 93ffc217 shell am instrument -w -e disableAnalytics true io.appium.uiautomator2.server.test/androidx.test.runner.AndroidJUnitRunner
        #   adb -L tcp:5037 fork-server server --reply-fd 4
        #   /Users/cloudchen/Library/Android/sdk/platform-tools/adb -P 5037 -s 93ffc217 logcat -v threadtime
        command = 'pkill -f appium'
        subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(command)
        time.sleep(1)  # 怕不停頓有機會造成錯誤

    def kill_adb_process(self):
        # 刪除 Appium 與 adb 留下的程序，不然執行 Appium 的 uiautomator2 會出錯
        # 程序長得像是這樣:
        #   /Users/cloudchen/Library/Android/sdk/platform-tools/adb -P 5037 -s 93ffc217 shell am instrument -w -e disableAnalytics true io.appium.uiautomator2.server.test/androidx.test.runner.AndroidJUnitRunner
        #   adb -L tcp:5037 fork-server server --reply-fd 4
        #   /Users/cloudchen/Library/Android/sdk/platform-tools/adb -P 5037 -s 93ffc217 logcat -v threadtime
        command = 'pkill -f adb'
        subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(command)
        time.sleep(1)  # 怕不停頓有機會造成錯誤

    # 關閉背景程序
    def close_background_app(self, d: u2):
        # HOME
        d.xpath('//*[@resource-id="com.android.systemui:id/center_group"]').click()

        # 清除背景程序
        d(resourceId="com.android.systemui:id/recent_apps").click()
        time.sleep(0.5)
        d(resourceId="com.oppo.launcher:id/clear_all_panel").click()
        time.sleep(0.5)

        # HOME
        d.xpath('//*[@resource-id="com.android.systemui:id/center_group"]').click()
        time.sleep(0.5)

    def setup_method(self, method):
        printlog("generated driver", is_debug=DEBUG)

    def teardown_method(self, method):
        printlog(json.dumps(self.http_response))

    def login(self, domain, username, password, login_success_element):
        self.page_driver.goto(f'{domain}/login')
        account_element = ((By.XPATH, '//input[@type="text"][@placeholder]'), "Account")
        password_element = ((By.XPATH, '//input[@type="password"]'), "Password")
        self.page_driver.sendkeys(username, account_element, wait_time=20)
        self.page_driver.sendkeys(password, password_element)
        self.page_driver.playwright_driver.keyboard.press('Enter')
        if 'm.' in domain:
            mb_announcement_close_btn = (
            (By.XPATH, '//*[@id="app"]/div[6]/div[2]/div[3]/div[@class="left"]'), 'MB關閉公告按鈕')
            self.page_driver.click_if_exist(mb_announcement_close_btn, wait_time=25)
        else:
            pc_announcement_close_btn = ((By.XPATH, '//body/div[3]/div/div[1]/button'), 'PC關閉公告按鈕')
            self.page_driver.click_if_exist(pc_announcement_close_btn, wait_time=25)
        self.page_driver.is_element_exist(login_success_element)

    def wait_until_image_on_screen(self, data: dict):
        """
        設定時間內不斷嘗試直到找到物件
        可指定是否要將用來判斷的 screenshot 存檔
        wait_time: 等待指定圖片出現的時間
        delay: 判斷到指定圖片後隔多久才進行截圖
        platform: 如果是 PC 的話會只擷取遊戲部分的圖片
        """
        template_name = data.get('template_name')
        template_name_2 = data.get('template_name_2')
        template_threshold = data.get('template_threshold', 0.5)
        wait_time = data.get('wait_time', 10)
        save_pic_name = data.get('save_pic_name', None)
        is_debug = data.get('is_debug', True)
        delay = data.get('delay', 0)
        crop = data.get('crop')
        # 預設連拍 1 張圖片
        continuous_screenshot = data.get('continuous_screenshot', 6)

        # TODO: 這一整段是在組出對應路徑的邏輯，可以抽出來外面組合好再丟進來
        # 傳進來的 template_name 用 # 切開 level 與 名稱，例：lv1#green_toolbox
        if "#" in template_name:
            level, template_img_name = template_name.split('#')
            template_path = os.path.join(self.template_image_root_path, template_img_name, f'{level}.jpg')
            if template_name_2 is not None and "#" in template_name_2:
                level_2, template_img_name_2 = template_name_2.split('#')
                template_path_2 = os.path.join(self.template_image_root_path, template_img_name_2, f'{level_2}.jpg')
            else:
                template_path_2 = os.path.join(self.template_image_root_path, f'{template_name_2}.jpg')
        else:
            template_path = os.path.join(self.template_image_root_path, f'{template_name}.jpg')
            template_path_2 = os.path.join(self.template_image_root_path, f'{template_name_2}.jpg')

        start_time = end_time = time.time()
        debug_dir = f"{self.screenshot_save_path}/debug"
        # ===========
        obj = None
        obj_2 = None
        time.sleep(delay)  # 睡幾秒再取圖片，e.g. 適用場景如判斷到倒數第1秒出現，過2秒後再進行截圖避免有動畫遮蔽畫面
        while (end_time - start_time < wait_time) and obj is None and obj_2 is None:
            screenshots_list = []
            # 預設連拍 3 張圖片
            for _ in range(continuous_screenshot):
                screenshot, img_b64 = self.page_driver.screenshot_to_base64()
                # 获取当前时间
                now = datetime.now()
                # 使用 strftime 函数将时间格式化为 HHMMSSss 格式
                formatted_time = now.strftime("%H%M%S%f")[:-4]
                screenshots_list.append((screenshot, img_b64, formatted_time))

            for index, scr_img_timestamp in enumerate(screenshots_list):
                screenshot, img_b64, timestamp = scr_img_timestamp
                if is_debug:
                    debug_path = os.path.join(debug_dir, f'{timestamp}_{template_name}_{index}.jpg')  # 指定保存的文件路径
                    cv2.imwrite(debug_path, screenshot)  # 保存debug图像
                obj = find_pic_x_y(img_b64, True, template_path, template_threshold)
                # 如果有餵第二張圖，會連帶一起比對，為了解決 screenshot 需要超過一秒的問題，給兩張圖一起比對避免錯過某一秒的截圖
                if template_name_2:
                    obj_2 = find_pic_x_y(img_b64, True, template_path_2, template_threshold)
                end_time = time.time()

                obj_debug_output_path = os.path.join(debug_dir,
                                                     f'{timestamp}_obj_1_{template_name}_{index}.jpg')  # 指定保存的文件路径
                obj_2_debug_output_path = os.path.join(debug_dir,
                                                       f'{timestamp}_obj_2_{template_name}_{index}.jpg')  # 指定保存的文件路径
                save_pic_path = os.path.join(self.screenshot_save_path, f'{save_pic_name}.jpg')
                if obj:
                    x, y, img = obj
                    if is_debug:
                        cv2.imwrite(obj_debug_output_path, img)  # 保存debug图像
                        # cv2.imshow('Matching Result', img)  # debug
                        # cv2.waitKey(0)  # debug
                        # cv2.destroyAllWindows()  # debug
                    if save_pic_name:
                        if crop:
                            x1, y1, width, height = crop
                            # x1 = 210
                            # y1 = 107
                            # width = 1500
                            # height = 840
                            # 設定要擷取的區域
                            roi = screenshot[y1:y1 + height, x1:x1 + width]
                            cv2.imwrite(save_pic_path, roi)  # 保存图像
                        else:
                            cv2.imwrite(save_pic_path, screenshot)  # 保存图像
                    break
                if obj_2:
                    x, y, img = obj_2
                    # 把拿來辨識用的圖存起來
                    if is_debug:
                        cv2.imwrite(obj_2_debug_output_path, img)  # 保存debug图像
                        # cv2.imshow('Matching Result', img)  # debug
                        # cv2.waitKey(0)  # debug
                        # cv2.destroyAllWindows()  # debug
                        # 如果有設定要順便存檔
                    if save_pic_name:
                        time.sleep(delay)  # 睡幾秒再取圖片，e.g. 適用場景如判斷到倒數第1秒出現，過2秒後再進行截圖避免有動畫遮蔽畫面
                        cv2.imwrite(save_pic_path, screenshot)  # 保存图像
                    break

        # 若透過兩張 template 圖都沒有找到東西，則報錯
        if obj is None and obj_2 is None:
            pic_file_name = os.path.split(template_path)[-1]
            if template_path_2:
                error_msg = f"{template_path}\n{template_path_2}\n沒找到 {pic_file_name} 物件"
            else:
                error_msg = f"{template_path}\n沒找到 {pic_file_name} 物件"
            raise RuntimeError(error_msg)
        else:
            return obj[:2] if obj else obj_2[:2]

    def wait_until_image_not_on_screen(self, data: dict, wait_time: float):
        start_time = time.time()
        while time.time() - start_time < wait_time:
            try:
                self.wait_until_image_on_screen(data)
                time.sleep(1)
                continue
            except RuntimeError:
                print(f'template_name: {data["template_name"]} is not on screen anymore')
                return
        raise RuntimeError(f'template_name: {data["template_name"]} still on screen')

    def find_images_on_screen(self, data: dict):
        """
        設定時間內不斷嘗試直到找到物件
        template_name: 用來批配檔案的檔名，以 # 分隔前面為 level 後面為 名字，例如：lv1#coin
        save_pic_name: 可指定是否要將用來判斷的 screenshot 存檔，有檔名會存檔，沒有則不存
        wait_time: 等待指定圖片出現的時間
        delay: 判斷到指定圖片後隔多久才進行截圖
        platform: 如果是 PC 的話會只擷取遊戲部分的圖片
        continuous_screenshot: 是否連續拍照後在做一次檢查裡面的圖片是否有符合的，用以辨識動圖用，預設為拍攝 6 張圖片
        """
        template_name = data.get('template_name')
        template_name_2 = data.get('template_name_2')
        template_threshold = data.get('template_threshold', 0.8)
        wait_time = data.get('wait_time', 10)
        save_pic_name = data.get('save_pic_name', None)
        is_debug = data.get('is_debug', True)
        delay = data.get('delay', 0)
        # 預設連拍 1 張圖片
        continuous_screenshot = data.get('continuous_screenshot', 6)

        # TODO: 這一整段是在組出對應路徑的邏輯，可以抽出來外面組合好再丟進來
        # 傳進來的 template_name 用 # 切開 level 與 名稱，例：lv1#green_toolbox
        if "#" in template_name:
            level, template_img_name = template_name.split('#')
            template_path = os.path.join(self.template_image_root_path, template_img_name, f'{level}.jpg')
            if template_name_2 is not None and "#" in template_name_2:
                level_2, template_img_name_2 = template_name_2.split('#')
                template_path_2 = os.path.join(self.template_image_root_path, template_img_name_2, f'{level_2}.jpg')
            else:
                template_path_2 = os.path.join(self.template_image_root_path, f'{template_name_2}.jpg')
        else:
            template_path = os.path.join(self.template_image_root_path, f'{template_name}.jpg')
            template_path_2 = os.path.join(self.template_image_root_path, f'{template_name_2}.jpg')

        start_time = end_time = time.time()
        debug_dir = f"{self.screenshot_save_path}/debug"
        # ===========
        objs = None
        objs_2 = None

        while (end_time - start_time < wait_time) and objs is None and objs_2 is None:
            screenshots_list = []
            # 預設連拍 3 張圖片
            for _ in range(continuous_screenshot):
                screenshot, img_b64 = self.page_driver.screenshot_to_base64()
                # 获取当前时间
                now = datetime.now()
                # 使用 strftime 函数将时间格式化为 HHMMSSss 格式
                formatted_time = now.strftime("%H%M%S%f")[:-4]
                screenshots_list.append((screenshot, img_b64, formatted_time))

            for index, scr_img_timestamp in enumerate(screenshots_list):
                screenshot, img_b64, timestamp = scr_img_timestamp
                if is_debug:
                    debug_path = os.path.join(debug_dir, f'{timestamp}_{template_name}_{index}.jpg')  # 指定保存的文件路径
                    cv2.imwrite(debug_path, screenshot)  # 保存debug图像
                objs = find_pics_x_y(img_b64, True, template_path, template_threshold)
                # 如果有餵第二張圖，會連帶一起比對，為了解決 screenshot 需要超過一秒的問題，給兩張圖一起比對避免錯過某一秒的截圖
                if template_name_2:
                    objs_2 = find_pics_x_y(img_b64, True, template_path_2, template_threshold)
                end_time = time.time()

                obj_debug_output_path = os.path.join(debug_dir,
                                                     f'{timestamp}_obj_1_{template_name}_{index}.jpg')  # 指定保存的文件路径
                obj_2_debug_output_path = os.path.join(debug_dir,
                                                       f'{timestamp}_obj_2_{template_name}_{index}.jpg')  # 指定保存的文件路径
                save_pic_path = os.path.join(self.screenshot_save_path, f'{save_pic_name}.jpg')
                if objs:
                    match_list, img = objs
                    if is_debug:
                        cv2.imwrite(obj_debug_output_path, img)  # 保存debug图像
                        # cv2.imshow('Matching Result', img)  # debug
                        # cv2.waitKey(0)  # debug
                        # cv2.destroyAllWindows()  # debug
                    if save_pic_name:
                        time.sleep(delay)  # 睡幾秒再取圖片，e.g. 適用場景如判斷到倒數第1秒出現，過2秒後再進行截圖避免有動畫遮蔽畫面
                        cv2.imwrite(save_pic_path, screenshot)  # 保存图像
                    break
                if objs_2:
                    match_list, img = objs_2
                    # 把拿來辨識用的圖存起來
                    if is_debug:
                        cv2.imwrite(obj_2_debug_output_path, img)  # 保存debug图像
                        # cv2.imshow('Matching Result', img)  # debug
                        # cv2.waitKey(0)  # debug
                        # cv2.destroyAllWindows()  # debug
                        # 如果有設定要順便存檔
                    if save_pic_name:
                        time.sleep(delay)  # 睡幾秒再取圖片，e.g. 適用場景如判斷到倒數第1秒出現，過2秒後再進行截圖避免有動畫遮蔽畫面
                        cv2.imwrite(save_pic_path, screenshot)  # 保存图像
                    break

        # 若透過兩張 template 圖都沒有找到東西，則報錯
        if objs is None and objs_2 is None:
            pic_file_name = os.path.split(template_path)[-1]
            if template_path_2:
                error_msg = f"{template_path}\n{template_path_2}\n沒找到 {pic_file_name} 物件"
            else:
                error_msg = f"{template_path}\n沒找到 {pic_file_name} 物件"
            raise RuntimeError(error_msg)
        else:
            return objs if objs else objs_2

    def read_number(self, image, crop_offset: tuple = None, ):
        """
        辨識圖片中的數字
        :param image:
        :param crop_offset: x1, y1, x2, y2  截取這區間的圖片
        :return:
        """
        # TODO: 設定傳進來的 image 型態
        # 讀取圖片並轉換為 Pillow 圖像對象
        # image = Image.open('/Users/cloudchen/Downloads/template.jpg')

        # 定義要擷取的區域座標
        x1, y1, x2, y2 = 240, 53, 340, 85

        # 設定語言模型為英文，並將配置設置為只辨識數字
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789:'

        if crop_offset:
            # 使用 crop 方法擷取特定區域的圖片
            cropped_image = image.crop((x1, y1, x2, y2))
            # cropped_image.show()  # debug

            # 使用 pytesseract 辨識圖像中的文本
            text = pytesseract.image_to_string(cropped_image, config=custom_config)

            print('辨識結果:', text)
            return text
        else:
            # 使用 pytesseract 辨識圖像中的文本
            text = pytesseract.image_to_string(image, config=custom_config)

            print('辨識結果:', text)
            return text

    def swipe_two_object(self, obj_1: tuple, obj_2: tuple):
        """
        傳進兩個物件，拖曳 obj_1 的位置到 obj_2 的位置
        obj 的格式為 find_images_on_screen 所回傳的 [((x, y), match), ... ] 其中兩個
        :param obj_1:
        :param obj_2:
        :return:
        """
        (x_1, y_1), match = obj_1  # match 批配度這邊用不到
        (x_2, y_2), match = obj_2
        self.page_driver.swipe(x_1, y_1, x_2, y_2)

    def click_by_image(self, template_name, offset: tuple = (0, 0), template_threshold: float = 0.75,
                       wait_time: float = 10, delay: float = 0.5):
        """
        餵一張圖片，判斷出圖片後得出圖片座標，並點擊圖片中心點位置
        template_path: 要點擊的圖片路徑
        offset: 要點擊偏移後的座標，有時候判斷跟要點擊的部分不一定一樣，e.g. 我想要點擊辨識到的圖片的左側 100px，offset=(-100, 0)
        template_threshold: 圖片相似度，最高 0.9，預設從 0.6 開始往上找
        wait_time: 等待圖片出現的時間，單位為秒
        """
        x, y = self.wait_until_image_on_screen({
            'template_name': template_name,
            'template_threshold': template_threshold,
            'wait_time': wait_time,
            'save_pic_name': '',  # 不傳入就不會儲存圖片
        })
        time.sleep(delay)  # 睡一下，避免後續動作過快造成問題
        printlog(f"於畫面上找到 {template_name} 並點擊", is_debug=DEBUG)
        self.page_driver.click_x_y(x, y, offset)
        time.sleep(1)  # 睡一下，避免後續動作過快造成問題
        return x, y

    def setup_stuff(self, setup_data: dict):
        """
        1. 依據語言/平台創建存截圖的資料夾
        2. 於平台切換指定語言
        """
        game_name = setup_data['game_name']
        self.screenshot_save_path = os.path.join(self.screenshot_root_save_path, game_name)
        # 設定儲存 Screenshot 的目錄
        Path(self.screenshot_save_path).mkdir(parents=True, exist_ok=True)
        # 設定 debug 圖片儲存路徑
        self.debug_dir = f"{self.screenshot_save_path}/debug"
        if os.path.exists(self.debug_dir):
            shutil.rmtree(self.debug_dir)  # 刪除舊資料夾
        Path(self.debug_dir).mkdir(parents=True, exist_ok=True)
        # 設定圖片比對時用來比對的圖片的資料夾根目錄
        self.template_image_root_path = os.path.join(TARGET_PICS_DIR, game_name)

    def dev_tool(self, item_count: int):
        first_row_y = 295
        next_row_offset = 230
        col_x = 735
        next_col_offset = 210

        # 第一排
        item_1 = (col_x, first_row_y)
        item_2 = (col_x - next_col_offset * 1, first_row_y)
        item_3 = (col_x - next_col_offset * 2, first_row_y)
        item_4 = (col_x - next_col_offset * 3, first_row_y)
        # 第二排
        item_5 = (col_x, first_row_y + next_row_offset * 1)
        item_6 = (col_x - next_col_offset * 1, first_row_y + next_row_offset * 1)
        item_7 = (col_x - next_col_offset * 2, first_row_y + next_row_offset * 1)
        item_8 = (col_x - next_col_offset * 3, first_row_y + next_row_offset * 1)
        # 第三排
        item_9 = (col_x, first_row_y + next_row_offset * 2)
        item_10 = (col_x - next_col_offset * 1, first_row_y + next_row_offset * 2)
        item_11 = (col_x - next_col_offset * 2, first_row_y + next_row_offset * 2)
        item_12 = (col_x - next_col_offset * 3, first_row_y + next_row_offset * 2)

        l = []
        for i in range(1, 13):
            l.append(eval(f'item_{i}'))
            if len(l) == item_count:
                break

        close_btn = self.wait_until_image_on_screen({
            'template_name': 'close_btn',
            'wait_time': 1,
        })
        close_btn_x, close_btn_y = close_btn

        for index, obj in enumerate(l, start=1):
            off_set_x, off_set_y = obj
            x = close_btn_x - off_set_x
            y = close_btn_y + off_set_y
            reset_point_x, reset_point_y = close_btn_x - 100, close_btn_y
            self.page_driver.click_x_y(reset_point_x, reset_point_y)
            self.page_driver.click_x_y(x, y)
            # crop
            crop = (213, 1008, 135, 120)
            self.wait_until_image_on_screen({
                'template_name': 'close_btn',
                'wait_time': 1,
                'delay': 1,
                'save_pic_name': f'lv{"max" if index == len(l) else index}',
                'crop': crop
            })

    @method_setup_teardown
    def test_fat_goose(self):
        """
        Fat Goose 肥額健身房
        """
        self.setup_stuff({
            'game_name': 'fat_goose',

        })

        # # 進入主程式後點擊閃電進入消消樂模式
        # self.click_by_image('main_game_btn', wait_time=120)
        # # 等待進入消消樂畫面
        # self.wait_until_image_on_screen({
        #     'template_name': 'back_btn',
        #     'wait_time': 1,
        # })

        # def find_obj_and_merge(obj_name: str, obj_level: int):
        #     template_name = f'lv{obj_level}#{obj_name}'
        #     # 找到物品
        #     match_list, img = self.find_images_on_screen({
        #         'template_name': template_name,
        #     })
        #     if len(match_list) >= 2:
        #         obj_1 = match_list[0]
        #         obj_2 = match_list[1]
        #         self.swipe_two_object(obj_1, obj_2)
        #     else:
        #         return

        self.wait_until_image_on_screen({
            'template_name': 'lv1#coin',
            'wait_time': 1,
        })
        print('debug')

    @method_setup_teardown
    def dev_fat_goose(self):
        """
        Fat Goose 肥額健身房
        """
        self.setup_stuff({
            'game_name': 'fat_goose',

        })
        self.dev_tool(item_count=6)


if __name__ == "__main__":
    start_time = time.time()

    with AndroidGame() as game:
        game.test_fat_goose()
        # game.dev_fat_goose()

    print(f'total test time: {time.time() - start_time}')
