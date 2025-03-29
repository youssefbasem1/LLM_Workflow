import os
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv('GROQ_API_KEY')
BASE_URL = os.getenv('BASE_URL')
LLM_MODEL = os.getenv('LLM_MODEL')

# Initialize the OpenAI client with custom base URL
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL 
)
# Define a function to make LLM API calls
def call_llm(messages, tools=None, tool_choice=None):
    """
    Make a call to the LLM API with the specified messages and tools.
    Args:
        messages: List of message objects
        tools: List of tool definitions (optional)
        tool_choice: Tool choice configuration (optional)
    Returns:
        The API response
    """
    kwargs = {
    "model": LLM_MODEL,
    "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools
    if tool_choice:
        kwargs["tool_choice"] = tool_choice
    try:
        response = client.chat.completions.create(**kwargs)
        return response
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return None
def get_sample_blog_post():
    """Read the sample blog post from a JSON file."""
    try:
        with open('sample_blog_post.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print("Error: sample_blog_post.json file not found.")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in sample_blog_post.json.")
        return None
extract_key_points_schema = {
    "type": "function",
    "function": {
        "name": "extract_key_points",
        "description": "Extract key points from a blog post",
        "parameters": {
            "type": "object",
            "properties": {
                "key_points": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of key points extracted from the blog post"
                }
            },
            "required": ["key_points"]
        }
    }
}
generate_summary_schema = {
    "type": "function",
    "function": {
        "name": "generate_summary",
        "description": "Generate a concise summary from the key points",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Concise summary of the blog post"
                }
            },
            "required": ["summary"]
        }
    }
}
create_social_media_posts_schema = {
    "type": "function",
    "function": {
        "name": "create_social_media_posts",
        "description": "Create social media posts for different platforms",
        "parameters": {
            "type": "object",
            "properties": {
                "twitter": {
                    "type": "string",
                    "description": "Post optimized for Twitter/X (max 280 characters)"
                },
                "linkedin": {
                    "type": "string",
                    "description": "Post optimized for LinkedIn (professional tone)"
                },
                "facebook": {
                    "type": "string",
                    "description": "Post optimized for Facebook"
                }
            },
            "required": ["twitter", "linkedin", "facebook"]
        }
    }
}
create_email_newsletter_schema = {
    "type": "function",
    "function": {
        "name": "create_email_newsletter",
        "description": "Create an email newsletter from the blog post and summary",
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "Email subject line"
                },
                "body": {
                    "type": "string",
                    "description": "Email body content in plain text"
                }
            },
            "required": ["subject", "body"]
        }
    }
}
def task_extract_key_points(blog_post):
    """
    Task function to extract key points from a blog post using tool calling.
    Args:
        blog_post: Dictionary containing the blog post
    Returns:
        List of key points
    """
    print(f"Debuging: blog_post type = {type(blog_post)}")  # Should print <class 'dict'>
    if not isinstance(blog_post, dict):
        raise TypeError("Expected blog_post to be a dictionary. Received:", type(blog_post))
    messages = [
        {"role": "system", "content": "You are an expert at analyzing content and extracting key points from articles."},
        {"role": "user", "content": f"Extract the key points from this blog post:\n\nTitle: {blog_post['title']}\n\nContent: {blog_post['content']}"}
    ]
    
    # Use tool_choice to ensure the model calls our function
    response = call_llm(
        messages=messages,
        tools=[extract_key_points_schema],
        tool_choice={"type": "function", "function": {"name": "extract_key_points"}}
    )
    
    # Extract the tool call information
    if response and response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        result = json.loads(tool_call.function.arguments)
        return result.get("key_points", [])
    return []
def task_generate_summary(key_points, max_length=150):
    """
    Task function to generate a concise summary from key points using tool calling.
    Args:
        key_points: List of key points extracted from the blog post
        max_length: Maximum length of the summary in words
    Returns:
        String containing the summary
    """
    messages = [
        {"role": "system", "content": "You are an expert at summarizing content concisely while preserving key information."},
        {"role": "user", "content": f"Generate a summary based on these key points, max {max_length} words:\n\n" +
         "\n".join([f"- {point}" for point in key_points])}
    ]
    
    # Use tool_choice to ensure the model calls our function
    response = call_llm(
        messages=messages,
        tools=[generate_summary_schema],
        tool_choice={"type": "function", "function": {"name": "generate_summary"}}
    )
    
    # Extract the tool call information
    if response and response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        result = json.loads(tool_call.function.arguments)
        return result.get("summary", "")
    return "" 
def task_create_social_media_posts(key_points, blog_title):
    """
    Task function to create social media posts for different platforms using tool calling.
    Args:
        key_points: List of key points extracted from the blog post
        blog_title: Title of the blog post
    Returns:
        Dictionary with posts for each platform
    """
    messages = [
        {"role": "system", "content": "You are a social media expert who creates engaging posts optimized for different platforms."},
        {"role": "user", "content": f"Create social media posts for Twitter, LinkedIn, and Facebook based on this blog title: '{blog_title}' and these key points:\n\n" +
         "\n".join([f"- {point}" for point in key_points])}
    ]
    
    # Use tool_choice to ensure the model calls our function
    response = call_llm(
        messages=messages,
        tools=[create_social_media_posts_schema],
        tool_choice={"type": "function", "function": {"name": "create_social_media_posts"}}
    )
    
    # Extract the tool call information
    if response and response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        return json.loads(tool_call.function.arguments)
    return {"twitter": "", "linkedin": "", "facebook": ""}
def task_create_email_newsletter(blog_post, summary, key_points):
    """
    Task function to create an email newsletter using tool calling.
    Args:
        blog_post: Dictionary containing the blog post
        summary: String containing the summary
        key_points: List of key points extracted from the blog post
    Returns:
        Dictionary with subject and body for the email newsletter
    """
    messages = [
        {"role": "system", "content": "You are an email marketing specialist who creates engaging newsletters."},
        {"role": "user", "content": f"Create an email newsletter based on this blog post:\n\nTitle: {blog_post['title']}\n\nSummary: {summary}\n\nKey Points:\n" +
         "\n".join([f"- {point}" for point in key_points])}
    ]
    
    # Use tool_choice to ensure the model calls our function
    response = call_llm(
        messages=messages,
        tools=[create_email_newsletter_schema],
        tool_choice={"type": "function", "function": {"name": "create_email_newsletter"}}
    )
    
    # Extract the tool call information
    if response and response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        return json.loads(tool_call.function.arguments)
    return {"subject": "", "body": ""}  # Fallback if tool calling fails
def run_pipeline_workflow(blog_post):
    """
    Run a simple pipeline workflow to repurpose content.
    Args:
        blog_post: Dictionary containing the blog post
    Returns:
        Dictionary with all the generated content
    """
    # Extract key points
    print("Extracting key points...")
    key_points = task_extract_key_points(blog_post)
    
    # Generate summary from key points
    print("Generating summary...")
    summary = task_generate_summary(key_points)
    
    # Create social media posts
    print("Creating social media posts...")
    social_posts = task_create_social_media_posts(key_points, blog_post['title'])
    
    # Create email newsletter
    print("Creating email newsletter...")
    email = task_create_email_newsletter(blog_post, summary, key_points)
    
    # Return all generated content
    return {
        "key_points": key_points,
        "summary": summary,
        "social_posts": social_posts,
        "email": email
    }
def run_dag_workflow(blog_post):
    """
    Run a DAG workflow to repurpose content.
    Args:
        blog_post: Dictionary containing the blog post
    Returns:
        Dictionary with all the generated content
    """
    # Start with extracting key points (this is still the first step)
    print("Extracting key points...")
    key_points = task_extract_key_points(blog_post)
    
    # Generate summary from key points
    print("Generating summary...")
    summary = task_generate_summary(key_points)
    
    # These two tasks can run in parallel since they don't depend on each other
    # In a real DAG system, you'd use async or threading to run these concurrently
    
    # Create social media posts (depends only on key_points)
    print("Creating social media posts...")
    social_posts = task_create_social_media_posts(key_points, blog_post['title'])
    
    # Create email newsletter (depends on blog_post, summary, and key_points)
    print("Creating email newsletter...")
    email = task_create_email_newsletter(blog_post, summary, key_points)
    
    return {
        "key_points": key_points,
        "summary": summary,
        "social_posts": social_posts,
        "email": email
    }
def extract_key_points_with_cot(blog_post):
    """
    Extract key points from a blog post using chain-of-thought reasoning.
    Args:
        blog_post: Dictionary containing the blog post
    Returns:
        List of key points
    """
    messages = [
        {"role": "system", "content": "You are an expert at analyzing content and extracting key points from articles."},
        {"role": "user", "content": f"I want you to extract the key points from this blog post. Before giving me the final list, think step-by-step about:\n\n1. What are the main themes of the article?\n2. What are the most important claims or arguments?\n3. What evidence or examples support these claims?\n4. What are the practical implications or takeaways?\n\nTitle: {blog_post['title']}\n\nContent: {blog_post['content']}"}
    ]
    
    # First, get the model to think through the content
    thinking_response = call_llm(messages=messages)
    
    if thinking_response:
        thinking = thinking_response.choices[0].message.content
        
        # Now ask it to extract the key points in a structured format
        messages.append({"role": "assistant", "content": thinking})
        messages.append({"role": "user", "content": "Based on your analysis, extract the key points as a structured list."})
        
        # Use tool_choice to ensure the model calls our function
        response = call_llm(
            messages=messages,
            tools=[extract_key_points_schema],
            tool_choice={"type": "function", "function": {"name": "extract_key_points"}}
        )
        
        # Extract the tool call information
        if response and response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            result = json.loads(tool_call.function.arguments)
            return result.get("key_points", [])
    
    # Fallback to the regular method if CoT fails
    return task_extract_key_points(blog_post)
def evaluate_content(content, content_type):
    """
    Evaluate the quality of generated content.
    Args:
        content: The content to evaluate
        content_type: The type of content (e.g., "summary", "social_media_post", "email")
    Returns:
        Dictionary with evaluation results and feedback
    """
    # Different prompts based on content type
    prompts = {
        "summary": "Evaluate this summary for clarity, conciseness, and completeness.",
        "social_media_post": "Evaluate these social media posts for engagement potential, clarity, and platform appropriateness.",
        "email": "Evaluate this email newsletter for reader value, engagement, clarity, and professionalism.",
        "key_points": "Evaluate these key points for comprehensiveness, clarity, and importance."
    }
    
    prompt = prompts.get(content_type, "Evaluate this content for quality and effectiveness.")
    
    # For different content types, format the content appropriately
    content_str = content
    if isinstance(content, list):
        content_str = "\n".join([f"- {item}" for item in content])
    elif isinstance(content, dict):
        content_str = "\n\n".join([f"{key.upper()}:\n{value}" for key, value in content.items()])
    
    messages = [
        {"role": "system", "content": "You are a content quality evaluator who provides objective assessments and constructive feedback."},
        {"role": "user", "content": f"{prompt}\n\nContent to evaluate:\n\n{content_str}\n\nProvide a quality score between 0 and 1, and specific feedback for improvement."}
    ]
    
    # Define the evaluation schema
    evaluation_schema = {
        "type": "function",
        "function": {
            "name": "evaluate_content",
            "description": "Evaluate content quality and provide feedback",
            "parameters": {
                "type": "object",
                "properties": {
                    "quality_score": {
                        "type": "number",
                        "description": "Quality score between 0 and 1"
                    },
                    "feedback": {
                        "type": "string",
                        "description": "Specific feedback for improvement"
                    }
                },
                "required": ["quality_score", "feedback"]
            }
        }
    }
    
    response = call_llm(
        messages=messages,
        tools=[evaluation_schema],
        tool_choice={"type": "function", "function": {"name": "evaluate_content"}}
    )
    
    if response and response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        return json.loads(tool_call.function.arguments)
    
    # Fallback if tool calling fails
    return {"quality_score": 0.5, "feedback": "Unable to evaluate content."}

def improve_content(content, feedback, content_type):
    """
    Improve content based on feedback.
    Args:
        content: The content to improve
        feedback: Feedback on how to improve the content
        content_type: The type of content
    Returns:
        Improved content
    """
    # Format the content appropriately based on type
    content_str = content
    if isinstance(content, list):
        content_str = "\n".join([f"- {item}" for item in content])
    elif isinstance(content, dict):
        content_str = "\n\n".join([f"{key.upper()}:\n{value}" for key, value in content.items()])
    
    messages = [
        {"role": "system", "content": "You are a content improvement specialist who makes targeted enhancements based on feedback."},
        {"role": "user", "content": f"Improve the following {content_type} based on this feedback:\n\nFeedback: {feedback}\n\nOriginal content:\n\n{content_str}"}
    ]
    
    # Define the schema based on content type
    schema = None
    tool_name = ""
    
    if content_type == "summary":
        schema = generate_summary_schema
        tool_name = "generate_summary"
    elif content_type == "social_media_post":
        schema = create_social_media_posts_schema
        tool_name = "create_social_media_posts"
    elif content_type == "email":
        schema = create_email_newsletter_schema
        tool_name = "create_email_newsletter"
    elif content_type == "key_points":
        schema = extract_key_points_schema
        tool_name = "extract_key_points"
    else:
        # Generic improvement schema
        schema = {
            "type": "function",
            "function": {
                "name": "improve_content",
                "description": f"Improve {content_type}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "improved_content": {
                            "type": "string",
                            "description": "Improved content"
                        }
                    },
                    "required": ["improved_content"]
                }
            }
        }
        tool_name = "improve_content"
    
    response = call_llm(
        messages=messages,
        tools=[schema],
        tool_choice={"type": "function", "function": {"name": tool_name}}
    )
    
    if response and response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        result = json.loads(tool_call.function.arguments)
        
        # Handle different response formats based on content type
        if content_type == "summary":
            return result.get("summary", content)
        elif content_type == "social_media_post":
            return result
        elif content_type == "email":
            return result
        elif content_type == "key_points":
            return result.get("key_points", content)
        else:
            return result.get("improved_content", content)
    
    return content  # Fallback if tool calling fails

def generate_with_reflexion(generator_func, max_attempts=3):
    """
    Apply Reflexion to a content generation function.
    Args:
        generator_func: Function that generates content
        max_attempts: Maximum number of correction attempts
    Returns:
        Function that generates self-corrected content
    """
    def wrapped_generator(*args, **kwargs):
        # Get the content type from kwargs or use a default
        content_type = kwargs.pop("content_type", "content")
        
        # Generate initial content
        content = generator_func(*args, **kwargs)
        
        print(f"Initial {content_type} generated. Evaluating quality...")
        
        # Evaluate and correct if needed
        for attempt in range(max_attempts):
            evaluation = evaluate_content(content, content_type)
            print(f"Quality score: {evaluation['quality_score']}")
            
            # If quality is good enough, return the content
            if evaluation["quality_score"] >= 0.8:  # Threshold for acceptable quality
                print(f"{content_type.capitalize()} is good quality. No further improvements needed.")
                return content
            
            # Otherwise, attempt to improve the content
            print(f"Attempting to improve {content_type} (attempt {attempt+1}/{max_attempts})...")
            print(f"Feedback: {evaluation['feedback']}")
            
            improved_content = improve_content(content, evaluation["feedback"], content_type)
            content = improved_content
        
        print(f"Final {content_type} after {max_attempts} improvement attempts.")
        return content
    
    return wrapped_generator

def run_workflow_with_reflexion(blog_post):
    """
    Run a workflow with Reflexion-based self-correction.
    Args:
        blog_post: Dictionary containing the blog post
    Returns:
        Dictionary with all the generated content
    """
    # Extract key points with reflexion
    print("Extracting key points with reflexion...")
    key_points = generate_with_reflexion(task_extract_key_points)(blog_post, content_type="key_points")
    
    # Generate summary with reflexion
    print("Generating summary with reflexion...")
    summary = generate_with_reflexion(task_generate_summary)(key_points, content_type="summary")
    
    # Create social media posts with reflexion
    print("Creating social media posts with reflexion...")
    social_posts = generate_with_reflexion(task_create_social_media_posts)(key_points, blog_post['title'], content_type="social_media_post")
    
    # Create email newsletter with reflexion
    print("Creating email newsletter with reflexion...")
    email = generate_with_reflexion(task_create_email_newsletter)(blog_post, summary, key_points, content_type="email")
    
    return {
        "key_points": key_points,
        "summary": summary,
        "social_posts": social_posts,
        "email": email
    }
def define_agent_tools():
    """
    Define the tools that the workflow agent can use.
    Returns:
        List of tool definitions
    """
    # Collect all previously defined tools
    all_tools = [
        extract_key_points_schema,
        generate_summary_schema,
        create_social_media_posts_schema,
        create_email_newsletter_schema
    ]
    
    # Add a "finish" tool
    finish_tool_schema = {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Complete the workflow and return the final results",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "The final summary"
                    },
                    "social_posts": {
                        "type": "object",
                        "properties": {
                            "twitter": {
                                "type": "string",
                                "description": "Post for Twitter/X"
                            },
                            "linkedin": {
                                "type": "string",
                                "description": "Post for LinkedIn"
                            },
                            "facebook": {
                                "type": "string",
                                "description": "Post for Facebook"
                            }
                        },
                        "description": "The social media posts for each platform"
                    },
                    "email": {
                        "type": "object",
                        "properties": {
                            "subject": {
                                "type": "string",
                                "description": "Email subject line"
                            },
                            "body": {
                                "type": "string",
                                "description": "Email body content"
                            }
                        },
                        "description": "The email newsletter"
                    }
                },
                "required": ["summary", "social_posts", "email"]
            }
        }
    }
    
    # Return all tools, including the finish tool
    return all_tools + [finish_tool_schema]

def execute_agent_tool(tool_name, arguments):
    """
    Execute a tool based on the tool name and arguments.
    Args:
        tool_name: The name of the tool to execute
        arguments: The arguments to pass to the tool
    Returns:
        The result of executing the tool
    """
    if tool_name == "extract_key_points":
        # Create a blog post structure if only title and content are provided
        if 'title' in arguments and 'content' in arguments:
            blog_post = {
                'title': arguments['title'],
                'content': arguments['content']
            }
            return {"key_points": task_extract_key_points(blog_post)}
        return {"error": "Missing required arguments for extract_key_points"}
    
    elif tool_name == "generate_summary":
        if 'key_points' in arguments:
            key_points = arguments['key_points']
            max_length = arguments.get('max_length', 150)
            return {"summary": task_generate_summary(key_points, max_length)}
        return {"error": "Missing required arguments for generate_summary"}
    
    elif tool_name == "create_social_media_posts":
        if 'key_points' in arguments and 'title' in arguments:
            key_points = arguments['key_points']
            title = arguments['title']
            return task_create_social_media_posts(key_points, title)
        return {"error": "Missing required arguments for create_social_media_posts"}
    
    elif tool_name == "create_email_newsletter":
        if all(k in arguments for k in ['blog_post', 'summary', 'key_points']):
            blog_post = arguments['blog_post']
            summary = arguments['summary']
            key_points = arguments['key_points']
            return task_create_email_newsletter(blog_post, summary, key_points)
        return {"error": "Missing required arguments for create_email_newsletter"}
    
    elif tool_name == "finish":
        # Just return the arguments as they contain the final results
        return arguments
    
    return {"error": f"Unknown tool: {tool_name}"}

def run_agent_workflow(blog_post):
    """
    Run an agent-driven workflow to repurpose content.
    Args:
        blog_post: Dictionary containing the blog post
    Returns:
        Dictionary with all the generated content
    """
    # Define the system message for the agent
    system_message = """
    You are a Content Repurposing Agent. Your job is to take a blog post and repurpose it into different formats:
    1. Extract key points from the blog post
    2. Generate a concise summary
    3. Create social media posts for different platforms
    4. Create an email newsletter
    
    You have access to tools that can help you with each of these tasks. Think carefully about which tools to use and in what order.
    
    For most efficient workflow:
    - First extract key points from the blog post
    - Use those key points to generate both the summary and social media posts
    - Use the blog post, key points, and summary to create the email newsletter
    
    When you're done, use the 'finish' tool to complete the workflow.
    """
    
    # Initialize the conversation
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Please repurpose this blog post:\n\nTitle: {blog_post['title']}\n\nContent: {blog_post['content']}"}
    ]
    
    # Define the agent tools
    tools = define_agent_tools()
    
    # Keep track of the results
    workflow_state = {
        "key_points": [],
        "summary": "",
        "social_posts": {},
        "email": {}
    }
    
    # Run the agent workflow
    print("Starting agent-driven workflow...")
    max_iterations = 10
    for i in range(max_iterations):
        print(f"Agent iteration {i+1}/{max_iterations}")
        
        # Get the agent's next action
        response = call_llm(messages, tools)
        
        if not response:
            print("Error: Failed to get a response from the LLM")
            break
        
        # Add the agent's response to the conversation
        messages.append(response.choices[0].message)
        
        # Check if the agent is done
        if not response.choices[0].message.tool_calls:
            print("Agent did not call any tools. Workflow complete.")
            break
        
        # Process the tool calls
        for tool_call in response.choices[0].message.tool_calls:
            # Extract tool information
            tool_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON in tool arguments: {tool_call.function.arguments}")
                continue
            
            print(f"Agent is using tool: {tool_name}")
            
            # Check if the agent is done
            if tool_name == "finish":
                print("Agent has completed the workflow.")
                return arguments
            
            # Execute the tool
            tool_result = execute_agent_tool(tool_name, arguments)
            
            # Update workflow state based on the tool result
            if tool_name == "extract_key_points" and "key_points" in tool_result:
                workflow_state["key_points"] = tool_result["key_points"]
            elif tool_name == "generate_summary" and "summary" in tool_result:
                workflow_state["summary"] = tool_result["summary"]
            elif tool_name == "create_social_media_posts":
                workflow_state["social_posts"] = tool_result
            elif tool_name == "create_email_newsletter":
                workflow_state["email"] = tool_result
            
            # Add the tool result to the conversation
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(tool_result) if isinstance(tool_result, dict) else tool_result
            })
    
    # If we reach here, the agent couldn't complete the workflow
    print("Warning: The agent couldn't complete the workflow within the maximum number of iterations.")
    
    # Return whatever we have in the workflow state
    return {
        "summary": workflow_state["summary"],
        "social_posts": workflow_state["social_posts"],
        "email": workflow_state["email"],
        "error": "The agent couldn't complete the workflow within the maximum number of iterations."
    }
def print_results(results):
    """
    Print the results of a workflow in a formatted way.
    Args:
        results: Dictionary with the workflow results
    """
    print("\n" + "=" * 60)
    print("  COMPARATIVE WORKFLOW RESULTS")
    print("=" * 60 + "\n")

    for approach, data in results.items():
        print(f"➡  {approach.upper()} WORKFLOW")
        print("-" * 60)
        
        print(f" Execution Time: {data['time_taken']:.2f} sec")
        print(f" Evaluation Metrics:")
        print(f"   - Coherence: {data['evaluation']['coherence']:.2f}")
        print(f"   - Relevance: {data['evaluation']['relevance']:.2f}")
        print(f"   - Completeness: {data['evaluation']['completeness']:.2f}\n")
        
        print(f" Key Points:")
        key_points = data["output"].get("key_points", [])
        for i, point in enumerate(key_points, 1):
            print(f"   {i}. {point}")
        
        print(f"\n Summary:\n   {data['output'].get('summary', 'N/A')[:300]}...\n")
        
        print(f" Social Media Posts:")
        for platform, post in data["output"].get("social_posts", {}).items():
            print(f"   - {platform.capitalize()}: {post[:200]}...")  # Truncate if too long
        
        print(f"\n Email:")
        email = data["output"].get("email", {})
        print(f"   - Subject: {email.get('subject', 'N/A')}")
        print(f"   - Body: {email.get('body', 'N/A')[:300]}...\n")  # Limit output
        
        print("-" * 60)

    print("\n Evaluation Complete!\n")

def evaluate_output(output, reference_summary):
    """
    Evaluate the generated output based on coherence, relevance, and completeness.
    Uses simple heuristics and LLM-based scoring if available.
    """
    scores = {"coherence": 0, "relevance": 0, "completeness": 0}

    # Coherence: Check if output is logically structured and readable
    scores["coherence"] = len(output.split()) / max(len(reference_summary.split()), 1)

    # Relevance: Check keyword overlap with the reference summary
    reference_words = set(reference_summary.lower().split())
    output_words = set(output.lower().split())
    scores["relevance"] = len(reference_words & output_words) / max(len(reference_words), 1)

    # Completeness: Check if the summary captures key points
    scores["completeness"] = 1 if len(output) >= 0.8 * len(reference_summary) else 0.5

    return scores


def compare_workflows(blog_post):
    """
    Runs the blog post through all three workflows and evaluates their outputs.
    Returns a comparative analysis.
    """
    results = {}

    # Run Pipeline Workflow
    start_time = time.time()
    pipeline_output = run_pipeline_workflow(blog_post)
    pipeline_time = time.time() - start_time

    # Run DAG Workflow with Reflexion
    start_time = time.time()
    dag_output = run_dag_workflow(blog_post)
    dag_time = time.time() - start_time

    # Run Agent-Driven Workflow
    start_time = time.time()
    agent_output = run_agent_workflow(blog_post)
    agent_time = time.time() - start_time

    # Use pipeline output as reference (or a manually written summary)
    reference_summary = pipeline_output.get("summary", "")

    # Evaluate Outputs
    results["pipeline"] = {
        "output": pipeline_output,
        "evaluation": evaluate_output(pipeline_output.get("summary", ""), reference_summary),
        "time_taken": pipeline_time,
    }

    results["dag"] = {
        "output": dag_output,
        "evaluation": evaluate_output(dag_output.get("summary", ""), reference_summary),
        "time_taken": dag_time,
    }

    results["agent"] = {
        "output": agent_output,
        "evaluation": evaluate_output(agent_output.get("summary", ""), reference_summary),
        "time_taken": agent_time,
    }

    return results


def generate_comparison_report(results):
    """
    Generates a detailed comparison report of all three approaches.
    """
    report = "\n=== Comparative Workflow Analysis ===\n"

    for approach, data in results.items():
        report += f"\n {approach.upper()} Workflow\n"
        report += f"    Execution Time: {data['time_taken']:.2f} sec\n"
        report += f"    Coherence Score: {data['evaluation']['coherence']:.2f}\n"
        report += f"    Relevance Score: {data['evaluation']['relevance']:.2f}\n"
        report += f"    Completeness Score: {data['evaluation']['completeness']:.2f}\n"
        report += f"    Summary Output: {data['output'].get('summary', 'N/A')[:200]}...\n"

    return report

def load_blog_post(file_path):
    """Loads the blog post from a JSON file and ensures it's a dictionary."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)  # Ensures it's a dictionary
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f" Error loading blog post: {e}")
        return None

def main():
    print("\n Running Comparative Evaluation on Blog Post...\n")
    
    # Load or define blog post
    raw_blog_post = """{
        "title": "How AI is Transforming Healthcare",
        "content": "Artificial Intelligence is revolutionizing healthcare..."
    }"""
    
    try:
        blog_post = json.loads(raw_blog_post) if isinstance(raw_blog_post, str) else raw_blog_post
    except json.JSONDecodeError:
        print(" Error: Invalid JSON format for blog post input.")
        return

    print(f" Debug: blog_post type = {type(blog_post)}\n")  # Debug check
    
    # Run comparative evaluation
    results = compare_workflows(blog_post)
    
    # Print formatted results
    print_results(results)

if __name__ == "__main__":
    main()