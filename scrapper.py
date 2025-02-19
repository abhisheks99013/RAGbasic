import os
import time
import asyncio
import trafilatura
from trafilatura.spider import focused_crawler
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor

# Set max links to crawl
MAX_LINKS = 20  # Adjust this as needed

# Initialize Selenium WebDriver
def init_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--window-size=1920x1080")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

# Extract text using Selenium for JavaScript-heavy sites
def extract_text_with_selenium(url):
    driver = init_driver()
    driver.get(url)
    
    # Wait for the page to load completely
    time.sleep(5)  # You can adjust this time or use WebDriverWait for specific elements

    page_source = driver.page_source
    soup = BeautifulSoup(page_source, "html.parser")
    text = soup.get_text(separator="\n", strip=True)

    # Extract all links from the page
    links = [urljoin(url, a["href"]) for a in soup.find_all("a", href=True)]

    driver.quit()
    return text, links

# Extract text using Trafilatura for static pages
async def extract_text_with_trafilatura(url):
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        extracted_text = trafilatura.extract(downloaded)
        return extracted_text if extracted_text else None
    return None

# Determine method and extract text
async def extract_text(url, visited_links, use_selenium=False):
    if use_selenium:
        print(f"Using Selenium for: {url}")
        text, links = extract_text_with_selenium(url)
    else:
        print(f"Using Trafilatura for: {url}")
        text = await extract_text_with_trafilatura(url)
        links = focused_crawler(url, max_seen_urls=5, max_known_urls=50)[1]

    if text:
        print(f"Extracted content from: {url}")
        print(text[:500] + "\n...\n")  # Print first 500 chars
    else:
        print(f"Failed to extract content from: {url}")

    # Recursively crawl new links
    for link in links:
        if link not in visited_links and len(visited_links) < MAX_LINKS:
            visited_links.add(link)
            await extract_text(link, visited_links, use_selenium=use_selenium)

# Entry point
async def main():
    homepage = "https://wiki.yudurobotics.com/index.php?title=Yudu_Robotics"
    
    # If site is JavaScript-heavy, use Selenium
    use_selenium = "linkedin.com" in homepage or "indiamart.com" in homepage  

    visited_links = set()
    await extract_text(homepage, visited_links, use_selenium=use_selenium)

if __name__ == "__main__":
    asyncio.run(main())