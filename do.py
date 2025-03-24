from playwright.sync_api import sync_playwright

def perform_action(page, action_json):
    action = action_json["action"]
    target_text = action_json["target_content"]
    value = action_json.get("value")

    element = page.locator(f"text={target_text}")

    if action == "click":
        print(f"Clicking: {target_text}")
        element.first.click()
    elif action == "type":
        print(f"Typing '{value}' into {target_text}")
        element.first.fill(value)
