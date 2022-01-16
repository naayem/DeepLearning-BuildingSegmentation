"""
This function takes the X and y swiss coordinates and convert it via the website 
https://tool-online.com/en/coordinate-converter.php into GPS coordinates
returning the latitude and the longitude
"""
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select

def swiss_to_gps(x,y):
    # Access to Chrome
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')

    web = webdriver.Chrome('chromedriver',chrome_options=chrome_options)
    web.get('https://tool-online.com/en/coordinate-converter.php')

    # Setup the input to be in swiss coordinates 
    ch = Select(web.find_element(By.XPATH,'//*[@id="Pays_src"]'))
    ch.select_by_value('Suisse')

    #Setup the output to be in GPS coordinates
    gps = Select(web.find_element(By.XPATH,'//*[@id="Projections_dest"]'))
    gps.select_by_value('EPSG:4326')

    # Send x and y values
    x_value = web.find_element(By.XPATH,'//*[@id="Coord1_src"]')
    x_value.send_keys(x)

    y_value = web.find_element(By.XPATH,'//*[@id="Coord2_src"]')
    y_value.send_keys(y)

    # Submit the input
    Submit = web.find_element(By.XPATH,'//*[@id="convertir"]')
    Submit.click()

    # Read the output
    getLat= web.find_element(By.CSS_SELECTOR,'#Coord2_dest')
    getLong= web.find_element(By.CSS_SELECTOR,'#Coord1_dest')

    return getLat.get_attribute('value'), getLong.get_attribute('value')
