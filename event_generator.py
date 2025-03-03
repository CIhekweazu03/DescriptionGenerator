import boto3
import json
from datetime import datetime
from typing import Dict, Any, Optional

class EventGenerator:
    """
    Generates professional event descriptions and volunteer expectations based on input form data.
    Uses AWS Bedrock's Claude model for natural language generation.
    """
    
    def __init__(
        self,
        model_id: str = 'anthropic.claude-3-sonnet-20240229-v1:0'
    ):
        """
        Initialize the generator with AWS Bedrock client.
        
        Args:
            model_id (str): The Bedrock model identifier to use
        """
        self.bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
        self.model_id = model_id
    
    def _format_datetime(self, date_str: str, time_str: str) -> str:
        """Format date and time strings into a readable format."""
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            time_obj = datetime.strptime(time_str, "%H:%M").time()
            
            # Format the date nicely
            date_formatted = date_obj.strftime("%A, %B %d, %Y")
            
            # Format the time with AM/PM
            time_formatted = time_obj.strftime("%I:%M %p").lstrip("0")
            
            return f"{date_formatted} at {time_formatted}"
        except Exception as e:
            print(f"Error formatting date/time: {e}")
            return f"{date_str} at {time_str}"
    
    def _get_event_duration(self, start_date: str, end_date: str, 
                           start_time: str, end_time: str) -> str:
        """Calculate and format the event duration."""
        try:
            start_datetime = datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M")
            end_datetime = datetime.strptime(f"{end_date} {end_time}", "%Y-%m-%d %H:%M")
            
            # Same day
            if start_date == end_date:
                start_time_formatted = start_datetime.strftime("%I:%M %p").lstrip("0")
                end_time_formatted = end_datetime.strftime("%I:%M %p").lstrip("0")
                date_formatted = start_datetime.strftime("%A, %B %d, %Y")
                return f"{date_formatted}, {start_time_formatted} to {end_time_formatted}"
            
            # Multi-day
            start_formatted = start_datetime.strftime("%A, %B %d, %Y at %I:%M %p").lstrip("0")
            end_formatted = end_datetime.strftime("%A, %B %d, %Y at %I:%M %p").lstrip("0")
            return f"{start_formatted} to {end_formatted}"
            
        except Exception as e:
            print(f"Error calculating duration: {e}")
            return f"{start_date} {start_time} to {end_date} {end_time}"
    
    def generate_event_description(self, event_data: Dict[str, Any]) -> str:
        """
        Generate a professional event description based on the provided form data.
        
        Args:
            event_data (Dict[str, Any]): Dictionary containing event information
            
        Returns:
            str: Generated event description
        """
        # Extract key information
        event_name = event_data.get("event_name", "")
        event_type = event_data.get("event_type", "")
        location_name = event_data.get("location_name", "")
        city = event_data.get("city", "")
        state = event_data.get("state", "")
        # Description is no longer an input field
        description = ""
        
        # Format time and dates
        time_info = self._get_event_duration(
            event_data.get("start_date", ""),
            event_data.get("end_date", ""),
            event_data.get("start_time", ""),
            event_data.get("end_time", "")
        )
        
        # Determine event venue description
        location_details = f"{location_name}, {city}, {state}"
        
        # Format for indoor/outdoor
        venue_type = event_data.get("venue_type", "")
        if venue_type:
            venue_description = f"This event will be held {venue_type.lower()}."
        else:
            venue_description = ""
            
        # Add the event category to provide more context
        event_category = event_data.get("event_category", "Standard Event")
            
        # Recurring event info
        recurring_info = ""
        if event_data.get("is_recurring", "No") == "Yes":
            recurring_dates = event_data.get("recurring_dates", [])
            if recurring_dates:
                dates_formatted = ", ".join(recurring_dates)
                recurring_info = f"This is a recurring event with additional dates: {dates_formatted}."
        
        # Create prompt for the LLM
        prompt = f"""
        Generate a professional, clear, and engaging event description based on the following information:
        
        Event Name: {event_name}
        Event Type: {event_type}
        Event Category: {event_category}
        Location: {location_details}
        When: {time_info}
        Venue Type: {venue_description}
        Recurring Information: {recurring_info}
        
        Guidelines:
        1. Start with a compelling introduction that clearly states the purpose of the event
        2. Include all essential details (what, when, where, who should attend)
        3. Highlight the benefits or value for participants
        4. Keep it concise (1-2 paragraphs)
        5. Use professional, engaging language
        6. Incorporate any specific details provided in the description field
        7. Make sure all logistical information is clearly presented
        
        Format the description as a cohesive, flowing narrative that would be appealing to potential attendees.
        Do not include headers or labels in the final description.
        """
        
        # Call the LLM
        try:
            response = self._invoke_model(prompt)
            return response.strip()
        except Exception as e:
            print(f"Error generating event description: {e}")
            return self._create_fallback_description(event_data)
    
    def _create_fallback_description(self, event_data: Dict[str, Any]) -> str:
        """Create a simple description if the LLM call fails."""
        event_name = event_data.get("event_name", "Our upcoming event")
        event_type = event_data.get("event_type", "")
        location = f"{event_data.get('location_name', '')}, {event_data.get('city', '')}, {event_data.get('state', '')}"
        
        time_info = self._get_event_duration(
            event_data.get("start_date", ""),
            event_data.get("end_date", ""),
            event_data.get("start_time", ""),
            event_data.get("end_time", "")
        )
        
        return f"""
        Join us for {event_name}, a {event_type} event. 
        
        This event will take place at {location} on {time_info}.
        
        {event_data.get('description', '')}
        
        We look forward to seeing you there!
        """
        
    def generate_volunteer_expectations(self, event_data: Dict[str, Any], event_description: str) -> str:
        """
        Generate volunteer expectations based on event data and description.
        
        Args:
            event_data (Dict[str, Any]): Dictionary containing event information
            event_description (str): The final event description
            
        Returns:
            str: Generated volunteer expectations
        """
        event_name = event_data.get("event_name", "")
        event_type = event_data.get("event_type", "")
        time_info = self._get_event_duration(
            event_data.get("start_date", ""),
            event_data.get("end_date", ""),
            event_data.get("start_time", ""),
            event_data.get("end_time", "")
        )
        
        # Create prompt for the LLM
        prompt = f"""
        Generate clear, detailed volunteer expectations for the following event:
        
        Event Name: {event_name}
        Event Type: {event_type}
        When: {time_info}
        Event Description: {event_description}
        
        Guidelines:
        1. Start with a brief introduction thanking volunteers
        2. Include specific arrival times (suggest 30-60 minutes before the event starts)
        3. Specify dress code appropriate for the event type (professional, business casual, etc.)
        4. List any items volunteers should bring
        5. Outline key responsibilities based on the event type
        6. Include information about breaks, meals, or refreshments
        7. Define the anticipated end time for volunteer duties
        8. Provide contact information placeholder for questions
        
        Format the volunteer expectations as a clear, bulleted list with section headers.
        Make sure the expectations are specific to the type of event.
        """
        
        # Call the LLM
        try:
            response = self._invoke_model(prompt)
            return response.strip()
        except Exception as e:
            print(f"Error generating volunteer expectations: {e}")
            return self._create_fallback_volunteer_expectations(event_data)
    
    def _create_fallback_volunteer_expectations(self, event_data: Dict[str, Any]) -> str:
        """Create simple volunteer expectations if the LLM call fails."""
        event_name = event_data.get("event_name", "the event")
        
        # Calculate arrival time (30 min before event)
        try:
            start_time = datetime.strptime(event_data.get("start_time", "09:00"), "%H:%M")
            arrival_time = (start_time - timedelta(minutes=30)).strftime("%I:%M %p").lstrip("0")
        except:
            arrival_time = "30 minutes before the event start time"
        
        return f"""
        # Volunteer Expectations for {event_name}
        
        ## Arrival and Check-in
        * Please arrive by {arrival_time} for check-in and briefing
        * Check in at the main entrance/registration desk
        
        ## Dress Code
        * Business casual attire is recommended
        * Wear comfortable shoes as you may be standing for extended periods
        
        ## Items to Bring
        * Water bottle
        * Name badge (if provided in advance)
        
        ## Responsibilities
        * Greet and direct participants
        * Assist with setup and breakdown
        * Support presenters and participants as needed
        
        ## Schedule
        * Volunteers should plan to stay until the conclusion of the event
        * Breaks will be coordinated by the volunteer coordinator
        
        ## Contact
        * For questions, please contact the volunteer coordinator
        """
    
    def _invoke_model(self, prompt: str) -> str:
        """Call the AWS Bedrock model with the given prompt."""
        try:
            # Prepare the request body
            request_body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "temperature": 0.5,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })
            
            # Make the API call
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                contentType='application/json',
                accept='application/json',
                body=request_body
            )
            
            # Parse the response
            response_body = json.loads(response['body'].read())
            content = response_body.get('content', [])
            
            if content and isinstance(content, list) and 'text' in content[0]:
                return content[0]['text']
            else:
                print("Unexpected response format from the model.")
                return ""
                
        except Exception as e:
            print(f"An error occurred while invoking the model: {e}")
            raise