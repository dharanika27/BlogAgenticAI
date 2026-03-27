from src.states.blogstate import BlogState
from langchain_core.messages import SystemMessage, HumanMessage
from src.states.blogstate import Blog

class BlogNode:
    """
    A class to represent he blog node
    """

    def __init__(self,llm):
        self.llm=llm

    
    def title_creation(self,state:BlogState):
        """
        create the title for the blog
        """

        if "topic" in state and state["topic"]:
            # Inside title_creation
            prompt = """
            You are an expert SEO copywriter. 
            Generate ONE creative and SEO-friendly blog title for the topic: {topic}.
            Return ONLY the title text. No conversational filler, no multiple options.
            """
            
            sytem_message=prompt.format(topic=state["topic"])
            print(sytem_message)
            response=self.llm.invoke(sytem_message)
            print(response)
            return {"blog":{"title":response.content}}
        
    def content_generation(self,state:BlogState):
        if "topic" in state and state["topic"]:
            system_prompt = """You are expert blog writer. Use Markdown formatting.
            Generate a detailed blog content with detailed breakdown for the {topic}"""
            system_message = system_prompt.format(topic=state["topic"])
            response = self.llm.invoke(system_message)
            return {
            "blog": {
                "title": state["blog"]["title"],
                "content": response.content
            }
            }
        
    def translation(self, state: BlogState):
        """
        Translate the content to the specified language using structured output.
        """
        translation_prompt = """
        You are a professional translator. Translate the following blog post into {current_language}.
        
        CRITICAL: You must return a valid JSON object containing:
        1. "title": The translated version of the title.
        2. "content": The translated version of the main content.

        ORIGINAL TITLE: {blog_title}
        ORIGINAL CONTENT: {blog_content}
        """
        
        # Extract current values from state
        blog_title = state["blog"]["title"]
        blog_content = state["blog"]["content"]
        target_lang = state["current_language"]

        messages = [
            HumanMessage(content=translation_prompt.format(
                current_language=target_lang, 
                blog_title=blog_title,
                blog_content=blog_content
            ))
        ]
        
        # Invoke the LLM with the structured schema
        try:
            translated_blog = self.llm.with_structured_output(Blog).invoke(messages)
            return {"blog": translated_blog}
        except Exception as e:
            print(f"Translation Error: {e}")
            # Fallback: if it fails, return the original state so the graph doesn't crash
            return {"blog": state["blog"]}

    def route(self, state: BlogState):
            return {"current_language": state['current_language'] }
    

    def route_decision(self, state: BlogState):
        """
        Route the content to the respective translation function.
        """
        if state["current_language"] == "hindi":
            return "hindi"
        elif state["current_language"] == "french": 
            return "french"
        else:
            return state['current_language']
        
        