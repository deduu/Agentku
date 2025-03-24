from settings import EMAIL, PASSWORD
import time
from look import parse_screen
from do import perform_action
from llm import get_action_from_user_input
from playwright.sync_api import sync_playwright

def agent_loop(user_input):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("https://www.google.com/")  # example login page
        time.sleep(5)

        screenshot_path = "screen.png"
        page.screenshot(path=screenshot_path, full_page=True)

        parsed_screen = parse_screen(screenshot_path)
        action_json = get_action_from_user_input(user_input, parsed_screen)

        if action_json.get("value") == "EMAIL":
            action_json["value"] = EMAIL
        elif action_json.get("value") == "PASSWORD":
            action_json["value"] = PASSWORD

        perform_action(page, action_json)

        time.sleep(10)
        browser.close()

if __name__ == "__main__":
    user_input = input("What do you want to do? ")
    agent_loop(user_input)

