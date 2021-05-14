from lib2to3.pgen2 import driver

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

my_driver = webdriver.Chrome('C:\Webdriver\chromedriver.exe', chrome_options=chrome_options)

my_driver.get("https://www.google.co.kr/imghp?hl=ko")

## 검색창 name="q"
element = my_driver.find_element_by_name("q")
element.send_keys("말")
element.send_keys(Keys.RETURN)

SCROLL_PAUSE_TIME = 1

## body 부분 스크롤
last_height = my_driver.execute_script("return document.body.scrollHeight")

while True:
    my_driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    time.sleep(SCROLL_PAUSE_TIME)

    new_height = my_driver.execute_script("return document.body.scroll")
    if new_height == last_height:
        try:
            ## html을 꾸며주는 css 그룹
            my_driver.find_element_by_css_selector(".mye4qd").click()
        except:
            break
    last_height = new_height

images = my_driver.find_element_by_css_selector(".rg_i Q4LuWd")
count = 1
for image in images:
    try:
        image.click()
        time.sleep(2)
        imgUrl = my_driver.find_element_by_xpath('//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div/div[2]/a/img').get_attribute("src")
        urllib.request.urlretrieve(imgUrl, "D:\GITHUB\ComputerVision\210506_data" + str(count) + ".jpg")
        count = count + 1

        ## 몇개나 다운 받을건지?
        if count == 20:
            break;
    except:
        pass

driver.close()

