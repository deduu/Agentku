import os
import time
import json
from typing import List, Dict, Any, Tuple, Optional
import base64
from PIL import Image
import io
import numpy as np
from settings import settings

# Import necessary libraries
try:
    from transformers import AutoModel
except ImportError:
    print("Transformers library not installed. To install, run: pip install transformers")

try:
    from openai import OpenAI
except ImportError:
    print("OpenAI library not installed or outdated. To install, run: pip install openai")

try:
    from playwright.sync_api import sync_playwright, Page, BrowserContext
except ImportError:
    print("Playwright not installed. To install, run: pip install playwright && playwright install")


class WebAgent:
    def __init__(self, openai_api_key: str, start_url: str = "https://www.google.com"):
        """
        Initialize the WebAgent with necessary components.
        
        Args:
            openai_api_key: API key for OpenAI
            start_url: Initial URL to navigate to
        """
        # Set up OpenAI
        self.openai_api_key = openai_api_key
        try:
            self.client = OpenAI(api_key=openai_api_key)
            print("OpenAI client initialized successfully")
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            self.client = None
        
        # Store start URL
        self.start_url = start_url
        
        # Initialize Playwright
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        
        # Initialize OmniParser model (if available)
        self.omniparser = None
        try:
            print("Loading OmniParser model...")
            self.omniparser = AutoModel.from_pretrained("microsoft/OmniParser-v2.0")
            print("OmniParser model loaded successfully")
        except Exception as e:
            print(f"Error loading OmniParser model: {e}")
            print("Using fallback parsing method")
    
    def start_browser(self, headless: bool = False) -> bool:
        """Start the browser session"""
        try:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=headless)
            self.context = self.browser.new_context()
            self.page = self.context.new_page()
            print("Browser started successfully")
            
            # Navigate to start URL
            success = self.navigate_to(self.start_url)
            if not success:
                print(f"Warning: Failed to navigate to start URL: {self.start_url}")
            
            return True
        except Exception as e:
            print(f"Error starting browser: {e}")
            return False
    
    def close_browser(self) -> None:
        """Close the browser session"""
        try:
            if self.page:
                self.page.close()
            if self.context:
                self.context.close()
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
            print("Browser closed successfully")
        except Exception as e:
            print(f"Error closing browser: {e}")
    
    def navigate_to(self, url: str) -> bool:
        """Navigate to a specific URL"""
        try:
            if not self.page:
                print("Browser not started")
                return False
                
            self.page.goto(url, timeout=30000)  # Increased timeout to 30 seconds
            print(f"Navigated to {url}")
            return True
        except Exception as e:
            print(f"Error navigating to {url}: {e}")
            return False
    
    def capture_screenshot(self) -> Optional[Image.Image]:
        """Capture a screenshot of the current page"""
        try:
            if not self.page:
                print("Browser not started")
                return None
                
            screenshot_bytes = self.page.screenshot()
            return Image.open(io.BytesIO(screenshot_bytes))
        except Exception as e:
            print(f"Error capturing screenshot: {e}")
            return None
    
    def parse_screen(self, screenshot: Optional[Image.Image] = None) -> List[Dict[str, Any]]:
        """
        Parse the screen using OmniParser to identify interactive elements
        
        Args:
            screenshot: Optional screenshot, if None will capture one
            
        Returns:
            List of parsed elements
        """
        try:
            if not self.page:
                print("Browser not started")
                return []
                
            if screenshot is None:
                screenshot = self.capture_screenshot()
                if screenshot is None:
                    return []
            
            # If using OmniParser model
            if self.omniparser:
                # Convert PIL Image to format expected by model
                screenshot_np = np.array(screenshot)
                
                # Process with OmniParser (placeholder - actual implementation depends on model API)
                try:
                    results = self.omniparser(screenshot_np)
                    parsed_elements = self._process_omniparser_results(results)
                    return parsed_elements
                except Exception as e:
                    print(f"Error parsing screen with OmniParser: {e}")
                    # Fall back to heuristic parsing
            
            # Fallback parsing using Playwright's accessibility tree
            return self._fallback_screen_parsing()
        except Exception as e:
            print(f"Error in parse_screen: {e}")
            return []
    
    def _process_omniparser_results(self, results) -> List[Dict[str, Any]]:
        """Process raw OmniParser results into standardized format"""
        # This would need to be adapted to the actual output format of OmniParser
        # Placeholder implementation
        parsed_elements = []
        
        try:
            # Example of processing results (adjust based on actual model output)
            for idx, element in enumerate(results.get('elements', [])):
                parsed_element = {
                    'id': idx,
                    'type': element.get('type', 'unknown'),
                    'bbox': element.get('bbox', [0, 0, 0, 0]),
                    'interactivity': element.get('interactivity', False),
                    'content': element.get('content', '')
                }
                parsed_elements.append(parsed_element)
        except Exception as e:
            print(f"Error processing OmniParser results: {e}")
        
        return parsed_elements
    
    def _fallback_screen_parsing(self) -> List[Dict[str, Any]]:
        """Fallback method to parse screen using Playwright's accessibility tree and selectors"""
        try:
            if not self.page:
                print("Browser not started")
                return []
            
            elements = []
            element_id = 0
            
            # Get all interactive elements using common selectors
            selectors = [
                'button', 'a', 'input', 'select', 'textarea', 
                '[role="button"]', '[role="link"]', '[role="checkbox"]',
                '[role="radio"]', '[role="textbox"]', '[role="combobox"]'
            ]
            
            for selector in selectors:
                try:
                    # Find elements matching this selector
                    locators = self.page.locator(selector).all()
                    
                    for locator in locators:
                        try:
                            # Get element properties
                            bbox = locator.bounding_box()
                            if not bbox:
                                continue
                                
                            # Normalize bounding box
                            viewport_size = self.page.viewport_size
                            normalized_bbox = [
                                bbox['x'] / viewport_size['width'],
                                bbox['y'] / viewport_size['height'],
                                (bbox['x'] + bbox['width']) / viewport_size['width'],
                                (bbox['y'] + bbox['height']) / viewport_size['height']
                            ]
                            
                            # Try to get text content
                            try:
                                content = locator.text_content() or ""
                                content = content.strip()
                            except:
                                content = ""
                                
                            # Try to get placeholder for inputs
                            if selector == 'input' or selector == 'textarea':
                                try:
                                    placeholder = locator.get_attribute('placeholder') or ""
                                    if placeholder and not content:
                                        content = f"Input field: {placeholder}"
                                except:
                                    pass
                                    
                            # Get element type
                            element_type = selector
                            if selector == 'input':
                                try:
                                    input_type = locator.get_attribute('type') or "text"
                                    element_type = f"input[{input_type}]"
                                except:
                                    pass
                            
                            # Add to elements list
                            elements.append({
                                'id': element_id,
                                'type': element_type,
                                'bbox': normalized_bbox,
                                'interactivity': True,
                                'content': content
                            })
                            element_id += 1
                            
                        except Exception as e:
                            print(f"Error processing element with selector {selector}: {e}")
                            continue
                            
                except Exception as e:
                    print(f"Error finding elements with selector {selector}: {e}")
                    continue
            
            # Also collect text elements for context
            try:
                # Find all paragraph and heading elements
                text_selectors = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'label', 'span', 'div']
                for selector in text_selectors:
                    try:
                        locators = self.page.locator(selector).all()
                        for locator in locators:
                            try:
                                content = locator.text_content() or ""
                                content = content.strip()
                                
                                if not content or len(content) < 2:
                                    continue
                                    
                                # Skip if this content is already in an interactive element
                                if any(content in elem['content'] for elem in elements):
                                    continue
                                    
                                bbox = locator.bounding_box()
                                if not bbox:
                                    continue
                                    
                                # Check if element is visible
                                is_visible = locator.is_visible()
                                if not is_visible:
                                    continue
                                    
                                # Normalize bounding box
                                viewport_size = self.page.viewport_size
                                normalized_bbox = [
                                    bbox['x'] / viewport_size['width'],
                                    bbox['y'] / viewport_size['height'],
                                    (bbox['x'] + bbox['width']) / viewport_size['width'],
                                    (bbox['y'] + bbox['height']) / viewport_size['height']
                                ]
                                
                                elements.append({
                                    'id': element_id,
                                    'type': selector,
                                    'bbox': normalized_bbox,
                                    'interactivity': False,
                                    'content': content
                                })
                                element_id += 1
                                
                            except Exception as e:
                                continue
                                
                    except Exception as e:
                        continue
            
            except Exception as e:
                print(f"Error collecting text elements: {e}")
            
            return elements
            
        except Exception as e:
            print(f"Error in fallback screen parsing: {e}")
            return []
    
    def click_element(self, element_identifier: Dict[str, Any]) -> bool:
        """
        Click on an element based on its identifier
        
        Args:
            element_identifier: Dict containing element information (could be content text, bbox, etc.)
            
        Returns:
            Success status
        """
        try:
            if not self.page:
                print("Browser not started")
                return False
                
            # Try different methods to click the element
            
            # Method 1: Try to click by text content
            if 'content' in element_identifier and element_identifier['content']:
                content = element_identifier['content'].strip()
                if content:
                    try:
                        # Try exact text match
                        self.page.click(f"text={content}")
                        print(f"Clicked element with text: {content}")
                        return True
                    except Exception:
                        try:
                            # Try partial text match with >> selector
                            self.page.click(f"text=>> {content}")
                            print(f"Clicked element containing text: {content}")
                            return True
                        except Exception:
                            pass
            
            # Method 2: Try to click by element type and content
            if 'type' in element_identifier and 'content' in element_identifier:
                elem_type = element_identifier['type']
                content = element_identifier['content'].strip()
                if content:
                    try:
                        # Construct a specific selector
                        if elem_type == 'button':
                            self.page.click(f"button:has-text('{content}')")
                            print(f"Clicked button with text: {content}")
                            return True
                        elif elem_type == 'a' or elem_type == 'link':
                            self.page.click(f"a:has-text('{content}')")
                            print(f"Clicked link with text: {content}")
                            return True
                        elif elem_type.startswith('input'):
                            # For input elements, try clicking by placeholder or nearby label
                            try:
                                self.page.click(f"input[placeholder='{content}']")
                                print(f"Clicked input with placeholder: {content}")
                                return True
                            except Exception:
                                try:
                                    self.page.click(f"label:has-text('{content}')")
                                    print(f"Clicked label with text: {content}")
                                    return True
                                except Exception:
                                    pass
                    except Exception:
                        pass
            
            # Method 3: Click by coordinates as last resort
            if 'bbox' in element_identifier and all(x is not None for x in element_identifier['bbox']):
                # Click by coordinates
                viewport_size = self.page.viewport_size
                x = int((element_identifier['bbox'][0] + element_identifier['bbox'][2]) / 2 * viewport_size['width'])
                y = int((element_identifier['bbox'][1] + element_identifier['bbox'][3]) / 2 * viewport_size['height'])
                
                self.page.mouse.click(x, y)
                print(f"Clicked element at coordinates: ({x}, {y})")
                return True
                
            print("Failed to click element - insufficient identifier information or element not found")
            return False
                
        except Exception as e:
            print(f"Error clicking element: {e}")
            return False
    
    def fill_input(self, element_identifier: Dict[str, Any], value: str) -> bool:
        """
        Fill an input field with a value
        
        Args:
            element_identifier: Dict containing element information
            value: Text to input
            
        Returns:
            Success status
        """
        try:
            if not self.page:
                print("Browser not started")
                return False
                
            # Try different methods to fill the input
            
            # Method 1: Try by content (which might be a label or placeholder)
            if 'content' in element_identifier and element_identifier['content']:
                content = element_identifier['content'].strip()
                if content:
                    try:
                        # Try by label text (looking for input near the label)
                        self.page.fill(f"input:near(:text('{content}'))", value)
                        print(f"Filled input near text '{content}' with value '{value}'")
                        return True
                    except Exception:
                        try:
                            # Try by placeholder
                            if "Input field:" in content:
                                # Extract placeholder from "Input field: {placeholder}"
                                placeholder = content.replace("Input field:", "").strip()
                                self.page.fill(f"input[placeholder='{placeholder}']", value)
                                print(f"Filled input with placeholder '{placeholder}' with value '{value}'")
                                return True
                            else:
                                self.page.fill(f"input[placeholder='{content}']", value)
                                print(f"Filled input with placeholder '{content}' with value '{value}'")
                                return True
                        except Exception:
                            pass
            
            # Method 2: Try to click element first, then type (useful for textareas, contenteditable divs, etc.)
            if self.click_element(element_identifier):
                try:
                    # Clear existing value if possible
                    try:
                        self.page.keyboard.press("Control+A")
                        self.page.keyboard.press("Delete")
                    except Exception:
                        pass
                        
                    # Type the value
                    self.page.keyboard.type(value)
                    print(f"Clicked element and typed '{value}'")
                    return True
                except Exception as e:
                    print(f"Error typing value after clicking: {e}")
                    return False
            
            # Method 3: Try to find by coordinates and type
            if 'bbox' in element_identifier and all(x is not None for x in element_identifier['bbox']):
                # Click by coordinates to focus
                viewport_size = self.page.viewport_size
                x = int((element_identifier['bbox'][0] + element_identifier['bbox'][2]) / 2 * viewport_size['width'])
                y = int((element_identifier['bbox'][1] + element_identifier['bbox'][3]) / 2 * viewport_size['height'])
                
                # Click to focus
                self.page.mouse.click(x, y)
                
                # Clear existing text if any
                try:
                    self.page.keyboard.press("Control+A")
                    self.page.keyboard.press("Delete")
                except Exception:
                    pass
                
                # Type the value
                self.page.keyboard.type(value)
                print(f"Clicked at coordinates ({x}, {y}) and typed '{value}'")
                return True
                
            print("Failed to fill input - insufficient identifier information or input not found")
            return False
                
        except Exception as e:
            print(f"Error filling input: {e}")
            return False
    
    def analyze_and_decide_action(self, task_description: str) -> Dict[str, Any]:
        """
        Use LLM to analyze current screen and decide next action
        
        Args:
            task_description: Description of the overall task
            
        Returns:
            Action to take
        """
        try:
            if not self.page or not self.client:
                print("Browser or OpenAI client not initialized")
                return {
                    "action_type": "wait",
                    "element_id": None,
                    "value": None,
                    "reasoning": "Browser or OpenAI client not initialized"
                }
            
            # Capture current state
            screenshot = self.capture_screenshot()
            if not screenshot:
                return {
                    "action_type": "wait",
                    "element_id": None,
                    "value": None,
                    "reasoning": "Failed to capture screenshot"
                }
                
            # Get current URL
            current_url = self.page.url
                
            # Parse the screen
            elements = self.parse_screen(screenshot)

            print(elements)
            
            if not elements:
                return {
                    "action_type": "wait",
                    "element_id": None,
                    "value": None,
                    "reasoning": "No elements detected on screen"
                }
            
            # Prepare elements for LLM (limit to important ones to save tokens)
            important_elements = []
            for elem in elements:
                if elem['interactivity'] or (elem['content'] and len(elem['content']) > 0):
                    important_elements.append(elem)
            
            # If still too many elements, prioritize interactive ones
            if len(important_elements) > 20:
                interactive_elements = [e for e in important_elements if e['interactivity']]
                text_elements = [e for e in important_elements if not e['interactivity'] and len(e['content']) > 5][:10]
                important_elements = interactive_elements + text_elements
            
            # Convert to string for LLM
            elements_str = json.dumps(important_elements, indent=2)
            
            # Prepare prompt for LLM
            prompt = f"""
            I'm looking at a web page with URL: {current_url}
            
            The page has the following key elements:
            
            {elements_str}
            
            My task is: {task_description}
            
            Based on the elements detected on the screen, what should I do next?
            Please provide your response in JSON format with these fields:
            1. "action_type": One of "click", "fill", "navigate", "scroll", "wait", or "complete"
            2. "element_id": The ID of the element to interact with (if applicable)
            3. "value": The value to fill (if action_type is "fill") or URL (if action_type is "navigate")
            4. "reasoning": Brief explanation of why this action
            
            If the task appears to be complete, return "action_type": "complete".
            If you need to wait for the page to load or update, return "action_type": "wait".
            
            Response should be valid JSON.
            """
            
            # Use OpenAI to analyze and suggest next action
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an AI assistant that helps navigate web pages to complete tasks."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            result_text = response.choices[0].message.content
            
            # Extract JSON from response
            try:
                result_text = result_text.strip()
                # If response is wrapped in code blocks, extract the JSON part
                if "```json" in result_text:
                    import re
                    json_match = re.search(r'```json\n(.*?)\n```', result_text, re.DOTALL)
                    if json_match:
                        result_text = json_match.group(1)
                
                action_json = json.loads(result_text)
                return action_json
                
            except json.JSONDecodeError as e:
                print(f"Error parsing LLM response as JSON: {e}")
                print(f"Raw response: {result_text}")
                
                # Fallback to a simple action
                return {
                    "action_type": "wait",
                    "element_id": None,
                    "value": None,
                    "reasoning": "Failed to parse LLM response"
                }
                
        except Exception as e:
            print(f"Error getting LLM recommendation: {e}")
            return {
                "action_type": "wait",
                "element_id": None,
                "value": None,
                "reasoning": f"Error: {str(e)}"
            }
    
    def execute_action(self, action: Dict[str, Any]) -> bool:
        """
        Execute the action recommended by the LLM
        
        Args:
            action: Action dict from analyze_and_decide_action
            
        Returns:
            Success status
        """
        try:
            if not self.page:
                print("Browser not started")
                return False
                
            action_type = action.get("action_type")
            element_id = action.get("element_id")
            value = action.get("value")
            
            print(f"Executing action: {action_type}")
            
            if action_type == "click" and element_id is not None:
                # Find the element by ID
                elements = self.parse_screen()
                target_element = next((e for e in elements if e['id'] == element_id), None)
                if target_element:
                    return self.click_element(target_element)
                else:
                    print(f"Element with ID {element_id} not found")
                    return False
                
            elif action_type == "fill" and element_id is not None and value is not None:
                # Find the element by ID
                elements = self.parse_screen()
                target_element = next((e for e in elements if e['id'] == element_id), None)
                if target_element:
                    return self.fill_input(target_element, value)
                else:
                    print(f"Element with ID {element_id} not found")
                    return False
                
            elif action_type == "navigate" and value is not None:
                return self.navigate_to(value)
                
            elif action_type == "scroll":
                try:
                    if value == "down":
                        self.page.evaluate("window.scrollBy(0, 300)")
                    elif value == "up":
                        self.page.evaluate("window.scrollBy(0, -300)")
                    elif value == "bottom":
                        self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    elif value == "top":
                        self.page.evaluate("window.scrollTo(0, 0)")
                    else:
                        self.page.evaluate("window.scrollBy(0, 300)")  # Default scroll down
                    print(f"Scrolled {value if value else 'down'}")
                    return True
                except Exception as e:
                    print(f"Error scrolling: {e}")
                    return False
                    
            elif action_type == "wait":
                wait_time = 2  # Default wait time in seconds
                if isinstance(value, int) and value > 0:
                    wait_time = min(value, 10)  # Cap at 10 seconds
                print(f"Waiting for {wait_time} seconds...")
                time.sleep(wait_time)
                return True
                
            elif action_type == "complete":
                print("Task completed!")
                return True
                
            else:
                print(f"Unknown action type: {action_type}")
                return False
                
        except Exception as e:
            print(f"Error executing action: {e}")
            return False
    
    def perform_task(self, task_description: str, max_steps: int = 10) -> bool:
        """
        Perform a complete task based on description
        
        Args:
            task_description: Natural language description of the task
            max_steps: Maximum number of steps to attempt
            
        Returns:
            Success status
        """
        print(f"Starting task: {task_description}")
        
        if not self.page:
            print("Browser not started")
            return False
        
        for step in range(max_steps):
            print(f"\nStep {step+1}/{max_steps}")
            
            # Analyze current state and decide on action
            action = self.analyze_and_decide_action(task_description)
            print(f"Recommended action: {action}")
            
            # Check if task is complete
            if action.get("action_type") == "complete":
                print(f"Task completed in {step+1} steps")
                return True
            
            # Execute action
            success = self.execute_action(action)
            if not success:
                print(f"Failed to execute action: {action}")
            
            # Give page time to update
            time.sleep(2)
        
        print(f"Maximum steps ({max_steps}) reached without completing task")
        return False


def run_example():
    """Example function to run the WebAgent for different tasks"""
    
    # Example task
    task = "Search for cheapest flight from Medan to Jakarta in March 2025"
    
    # Replace with your actual OpenAI API key
    openai_api_key = settings.OPENAI_API_KEY
    
    print("\n--- Starting Web Agent ---\n")
    
    # Initialize the agent
    agent = WebAgent(
        openai_api_key=openai_api_key,
        start_url="https://www.google.com"
    )
    
    try:
        # Start browser (with UI visible)
        if not agent.start_browser(headless=False):
            print("Failed to start browser. Exiting.")
            return
        
        # Perform the task
        success = agent.perform_task(task, max_steps=15)
        
        if success:
            print("\n--- Task completed successfully! ---")
        else:
            print("\n--- Task did not complete within the allowed steps ---")
            
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        # Always clean up
        agent.close_browser()
        print("\n--- Web Agent stopped ---")


if __name__ == "__main__":
    run_example()


from openai import OpenAI

print(OPENAI_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": "Write a one-sentence bedtime story about a unicorn."
    }]
)

print(completion.choices[0].message.content)

