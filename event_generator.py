import boto3
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

class EventGenerator:
    """
    Generates professional event descriptions and volunteer expectations based on input form data.
    Enhanced with improved prompts for more specific, audience-focused content.
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
        
        # Example descriptions for different event types to help guide generation
        self.example_descriptions = {
            "Career Fair": """
            NIWC Atlantic is hosting a Career Fair at Charleston Southern University. This event will showcase career opportunities and allow students to interact with professionals from various fields. Students will have the opportunity to learn about different career paths, required skills, and education requirements. Representatives will be available to discuss internship and job opportunities.
            """,
            
            "STEM Activity": """
            NIWC Atlantic regularly hosts targeted STEM Activity sessions. For the Spring of 2025, the activity is Life Budgeting. Life Budgeting is a scenario-based activity where students are randomly assigned a career, salary, family, and credit score. They must use that scenario to make financial decisions such as housing and transportation. Life budgeting is held within a classroom setting, with NIWC Atlantic professionals teaching the curriculum, and is designed to fit within a standard class period. Multiple sessions may be taught in a day.
            """,
            
            "Workshop": """
            Join us for our hands-on Robotics Workshop, designed to introduce students to the fundamentals of robotics engineering. Participants will learn about mechanical design, programming, and problem-solving while building and programming their own robot. This interactive workshop is suitable for students with no prior robotics experience and will give them insight into STEM career pathways.
            """
        }
        
        # Example volunteer expectations for different event types
        self.example_volunteer_expectations = {
            "Career Fair": """
            # Volunteer Expectations - Career Fair

            ## Arrival and Check-in
            * Please arrive by 8:00 AM (1 hour before the event starts) for setup and briefing
            * Check in at the Registration Desk in the main lobby
            * Wear your NIWC Atlantic badge for identification



            ## Items to Bring
            * Business cards
            * Prepared talking points about your role and department
            * Any demonstration materials you've been assigned

            ## Responsibilities
            * Staff your assigned booth for the duration of the event
            * Engage with students and discuss career opportunities
            * Collect resumes from interested candidates
            * Answer questions about NIWC Atlantic and your specific role
            * Assist with booth setup and breakdown

            ## Schedule
            * 8:00 AM - 9:00 AM: Setup and volunteer briefing
            * 9:00 AM - 2:00 PM: Career Fair (with rotating lunch breaks)
            * 2:00 PM - 2:30 PM: Breakdown and cleanup

            ## Contact
            * For questions before the event, contact the Volunteer Coordinator at volunteer.coordinator@example.mil
            * Day-of event questions: Call or text Event Lead at (555) 123-4567
            """,
            
            "STEM Activity": """
            # Volunteer Expectations - Life Budgeting STEM Activity

            ## Preparation
            * Watch the training video before the event (NWA may be used for this time)
            * Familiarize yourself with the Life Budgeting curriculum and materials
            * Print and prepare materials if not already provided:
              - Bio Cards
              - Family Cards
              - Credit Cards
              - Surprise Cards

            ## Arrival and Check-in
            * Arrive at the school 30 minutes before your first scheduled class
            * Check in at the front office to receive a visitor badge
            * Report to the classroom 15 minutes before class starts



            ## Items to Bring
            * Printed materials (if not provided by the school)
            * Laptop (if needed for presentations)
            * Water bottle and snacks for between sessions

            ## Class Schedule
            * 8:05 AM – 8:55 AM: Period 1
            * 9:00 AM – 9:50 AM: Period 2
            * 9:55 AM - 10:57 AM: Period 3
            * 11:00 AM - 12:02 PM: Period 4
            * 12:07 PM – 12:52 PM: Lunch Break
            * 12:57 PM – 1:57 PM: Period 5
            * 2:00 PM - 3:00 PM: Period 6

            ## Responsibilities
            * Deliver the Life Budgeting curriculum to each assigned class
            * Manage student groups and activity flow
            * Assist students with understanding financial concepts
            * Maintain classroom discipline in partnership with the teacher
            
            ## Important Notes
            * Make sure to charge hours to RG on your timecard when supporting this activity (STEM events are considered in-the-office)
            * Lunch will not be provided; plan accordingly
            
            ## Contact
            * For questions about the curriculum: education.coordinator@example.mil
            * For day-of logistics: event.lead@example.mil or (555) 987-6543
            """
        }
    
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
    
    def _get_relevant_example(self, event_type: str) -> str:
        """Find the most relevant example description based on event type."""
        # Convert event type to lowercase for matching
        event_type_lower = event_type.lower()
        
        # Check for partial matches in the example descriptions
        for key, example in self.example_descriptions.items():
            if key.lower() in event_type_lower:
                return example
        
        # If no match found, try to find partial matches
        for key, example in self.example_descriptions.items():
            if any(word in event_type_lower for word in key.lower().split()):
                return example
        
        # Default to a general example or the first one if no matches
        return list(self.example_descriptions.values())[0] if self.example_descriptions else ""
    
    def _get_relevant_volunteer_example(self, event_type: str) -> str:
        """Find the most relevant volunteer expectations example based on event type."""
        # Convert event type to lowercase for matching
        event_type_lower = event_type.lower()
        
        # Check for matches in the example volunteer expectations
        for key, example in self.example_volunteer_expectations.items():
            if key.lower() in event_type_lower:
                return example
        
        # If no match found, try to find partial matches
        for key, example in self.example_volunteer_expectations.items():
            if any(word in event_type_lower for word in key.lower().split()):
                return example
        
        # Default to a general example or the first one if no matches
        return list(self.example_volunteer_expectations.values())[0] if self.example_volunteer_expectations else ""
    
    def _get_target_audience(self, event_type: str) -> str:
        """Infer the likely target audience based on event type."""
        event_type_lower = event_type.lower()
        
        if any(term in event_type_lower for term in ["career fair", "job", "internship"]):
            return "students and job seekers"
        elif any(term in event_type_lower for term in ["stem", "science", "math", "cyber", "robotics", "camp"]):
            return "students and educators"
        elif any(term in event_type_lower for term in ["workshop", "training", "professional"]):
            return "professionals and educators"
        else:
            return "participants"
    
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
        street_address = event_data.get("street_address", "")
        city = event_data.get("city", "")
        state = event_data.get("state", "")
        
        # Format time and dates
        time_info = self._get_event_duration(
            event_data.get("start_date", ""),
            event_data.get("end_date", ""),
            event_data.get("start_time", ""),
            event_data.get("end_time", "")
        )
        
        # Format location details
        location_parts = [p for p in [location_name, street_address, city, state] if p]
        location_details = ", ".join(location_parts)
        
        # Determine event venue description
        venue_type = event_data.get("venue_type", "")
        venue_description = f"This event will be held {venue_type.lower()}." if venue_type else ""
            
        # Add the event category for context
        event_category = event_data.get("event_category", "Standard Event")
            
        # Recurring event info
        recurring_info = ""
        if event_data.get("is_recurring", "No") == "Yes":
            recurring_dates = event_data.get("recurring_dates", [])
            if recurring_dates:
                dates_formatted = ", ".join(recurring_dates)
                recurring_info = f"This is a recurring event with additional dates: {dates_formatted}."
        
        # Get relevant example
        example_description = self._get_relevant_example(event_type)
        
        # Infer target audience
        target_audience = self._get_target_audience(event_type)
        
        # Create enhanced prompt for the LLM
        prompt = f"""
        Generate a professional, clear, and engaging event description for a {event_type} based on the following information:
        
        Event Name: {event_name}
        Event Type: {event_type}
        Event Category: {event_category}
        Location: {location_details}
        When: {time_info}
        Venue Type: {venue_description}
        Recurring Information: {recurring_info}
        Target Audience: {target_audience}
        
        Here's an example of a good description for a similar type of event:
        {example_description}
        
        Guidelines:
        1. Begin with a strong opening that clearly states what the event is and its purpose
        2. Include all essential details (what participants will do/learn, when, where, who should attend)
        3. Highlight specific benefits for the target audience
        4. Keep it concise (2-3 paragraphs maximum)
        5. Use professional, engaging language appropriate for a government/military organization
        6. For STEM events, clearly explain the activities students will participate in
        7. For career events, emphasize networking and professional development opportunities
        8. Include any special requirements (materials needed, prerequisites, etc.)
        9. End with a clear call to action (registration information, contact details, etc.)
        10. If applicable, mention that the event is hosted by NIWC Atlantic
        
        Format the description as a cohesive, flowing narrative without headers or labels.
        Focus on being clear, specific, and informative while maintaining a professional tone.
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
        
        This event will take place at {location} on {time_info}. Participants will have the opportunity to engage in activities related to {event_type}.
        
        For more information or to register, please contact the event coordinator.
        """
        
    def generate_volunteer_expectations(self, event_data: Dict[str, Any], event_description: str) -> str:
        """
        Generate detailed volunteer expectations based on event data and description.
        
        Args:
            event_data (Dict[str, Any]): Dictionary containing event information
            event_description (str): The final event description
            
        Returns:
            str: Generated volunteer expectations
        """
        event_name = event_data.get("event_name", "")
        event_type = event_data.get("event_type", "")
        
        # Calculate arrival time (30-60 min before event)
        arrival_time = ""
        try:
            start_date = event_data.get("start_date", "")
            start_time = event_data.get("start_time", "")
            if start_date and start_time:
                start_datetime = datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M")
                arrival_datetime = start_datetime - timedelta(minutes=45)  # Default to 45 min before
                arrival_time = arrival_datetime.strftime("%I:%M %p").lstrip("0")
        except Exception:
            arrival_time = "45 minutes before the event start time"
        
        # Format the event time information
        time_info = self._get_event_duration(
            event_data.get("start_date", ""),
            event_data.get("end_date", ""),
            event_data.get("start_time", ""),
            event_data.get("end_time", "")
        )
        
        # Determine if it's a single day or multi-day event
        is_multi_day = event_data.get("start_date") != event_data.get("end_date")
        
        # Get relevant volunteer expectations example
        example_expectations = self._get_relevant_volunteer_example(event_type)
        
        # Create enhanced prompt for the LLM
        prompt = f"""
        Generate detailed, specific volunteer expectations for the following event:
        
        Event Name: {event_name}
        Event Type: {event_type}
        When: {time_info}
        Suggested Volunteer Arrival Time: {arrival_time}
        Multi-day Event: {"Yes" if is_multi_day else "No"}
        
        Event Description: 
        {event_description}
        
        Here's an example of good volunteer expectations for a similar type of event:
        {example_expectations}
        
        Guidelines:
        1. Format as a well-organized document with clear section headers using markdown (# for main headers, ## for subheaders)
        2. Begin with a brief, appreciative introduction thanking volunteers
        3. Include these specific sections:
           - Preparation: Any pre-event training or preparation required
           - Arrival and Check-in: Specify exact arrival time (typically 45 minutes before event start) and check-in location

           - Items to Bring: List specific items volunteers should bring
           - Responsibilities: Detail specific tasks volunteers will be expected to perform
           - Schedule: Provide a clear timeline of the volunteer duties, including breaks
           - Contact Information: Include placeholder for coordinator contact details
        4. For STEM events, include any curriculum details or classroom-specific guidance
        5. For multi-day events, specify expectations for each day
        6. Match the formality and tone to a government/military organization
        7. Include any special instructions related to the specific venue or event type
        8. If applicable, note how volunteers should record their hours (e.g., timecard codes)
        
        Be specific and practical - these instructions need to provide clear guidance for volunteers.
        Focus on concrete details rather than general statements.
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
        
        # Calculate arrival time (45 min before event)
        try:
            start_time = datetime.strptime(event_data.get("start_time", "09:00"), "%H:%M")
            arrival_time = (start_time - timedelta(minutes=45)).strftime("%I:%M %p").lstrip("0")
        except:
            arrival_time = "45 minutes before the event start time"
        
        return f"""
        # Volunteer Expectations for {event_name}
        
        Thank you for volunteering for this event. Your support is essential to making it a success!
        
        ## Arrival and Check-in
        * Please arrive by {arrival_time} for check-in and briefing
        * Check in at the main entrance/registration desk
        * A volunteer coordinator will provide you with any necessary materials and instructions
        

        
        ## Items to Bring
        * Government ID/badge
        * Water bottle
        * Pen and notepad
        
        ## Responsibilities
        * Greet and direct participants
        * Assist with setup and breakdown
        * Support presenters and participants as needed
        * Other duties as assigned by the volunteer coordinator
        
        ## Schedule
        * Volunteers should plan to stay until the conclusion of the event
        * Breaks will be coordinated by the volunteer coordinator
        
        ## Contact
        * For questions before the event, please contact the volunteer coordinator
        * On the day of the event, report to the volunteer check-in desk
        """
    
    def _invoke_model(self, prompt: str) -> str:
        """Call the AWS Bedrock model with the given prompt."""
        try:
            # Prepare the request body
            request_body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "temperature": 0.4,  # Slightly lower temperature for more consistent outputs
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