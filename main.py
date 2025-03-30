# Install required packages
# !pip install gradio sentence-transformers scikit-learn nltk

import re
import gradio as gr
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# Download NLTK resources
nltk.download(['stopwords', 'wordnet', 'vader_lexicon'])

class TicketData:
    """Class to hold customer support ticket data"""
    def __init__(self, message, industry="general", priority="medium", language="en"):
        self.message = message
        self.industry = industry.lower()
        self.priority = priority.lower()
        self.language = language
        self.sentiment = self._analyze_sentiment()

    def _analyze_sentiment(self):
        """Analyze message sentiment using VADER"""
        sia = SentimentIntensityAnalyzer()
        return sia.polarity_scores(self.message)['compound']

class AutoResponder:
    """AI-powered automatic response generator with SBERT embeddings"""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.industry_templates = self._load_templates()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.training_data, self.labels = self._prepare_training_data()
        self.embeddings = self._generate_embeddings()

    def _prepare_training_data(self):
        """Enhanced training data for intent classification"""
        sample_texts = [
            # Technical Issues
            "login problem", "can't access account", "password reset not working",
            "website down", "error 404", "connection failed",
            
            # Billing/Payments
            "payment declined", "invoice discrepancy", "refund status",
            "subscription cancellation", "overcharge dispute",
            
            # Shipping/Orders
            "package not delivered", "order tracking", "missing items",
            "damaged goods", "delayed shipment",
            
            # General Inquiries
            "account information", "service hours", "store location",
            "product availability", "price check"
        ]
        
        sample_labels = [
            "technical", "technical", "technical",
            "technical", "technical", "technical",
            "billing", "billing", "billing",
            "billing", "billing",
            "shipping", "shipping", "shipping",
            "shipping", "shipping",
            "general", "general", "general",
            "general", "general"
        ]
        
        return sample_texts, sample_labels

    def _generate_embeddings(self):
        """Generate sentence embeddings for training data"""
        return self.model.encode(self.training_data)

    def _load_templates(self):
        """Load enhanced response templates with fallbacks"""
        return {
            "tech": {
                "technical": {
                    "high": "Urgent technical issue detected (Priority: {priority}). Our engineering team is investigating immediately.",
                    "medium": "Technical support request received. We'll respond within 2 business hours.",
                    "low": "Thank you for your technical inquiry. We'll follow up within 24 hours."
                },
                "billing": {
                    "high": "Urgent billing matter escalated to finance team (Priority: {priority})",
                    "medium": "Payment issue received. Will resolve within 3 business days.",
                    "low": "Billing inquiry noted. We'll respond within 5 business days."
                },
                "default": "Technical support request received. Case #{random_id} created."
            },
            "retail": {
                "shipping": {
                    "high": "Urgent shipping issue identified (Priority: {priority}). Logistics team notified.",
                    "medium": "Delivery concern received. Investigating with carrier.",
                    "low": "Shipping inquiry logged. Expect response within 48 hours."
                },
                "billing": {
                    "high": "Retail billing escalation created (Priority: {priority})",
                    "medium": "Retail payment issue received. Processing resolution.",
                    "low": "Retail billing question noted. Response within 3 days."
                },
                "default": "Retail customer service request received. Case #{random_id}."
            },
            "general": {
                "default": {
                    "high": "Urgent support request received (Priority: {priority}). Team notified.",
                    "medium": "Support request received. Response expected within 24 hours.",
                    "low": "Inquiry logged. We'll respond within 3 business days."
                }
            }
        }

    def _preprocess_text(self, text):
        """Enhanced text preprocessing"""
        text = re.sub(r'[^\w\s!?]', '', text.lower())
        tokens = [self.lemmatizer.lemmatize(word) 
                  for word in text.split() if word not in self.stop_words]
        return ' '.join(tokens)

    def _classify_intent_with_embedding(self, text):
        """Classify intent using sentence embeddings and cosine similarity"""
        cleaned_text = self._preprocess_text(text)
        text_embedding = self.model.encode([cleaned_text])
        
        similarities = cosine_similarity(text_embedding, self.embeddings)
        best_match_idx = similarities.argmax()
        
        return self.labels[best_match_idx]

    def _get_response_template(self, intent, industry, priority):
        """Robust template selection with multiple fallbacks"""
        industry = industry if industry in self.industry_templates else "general"
        priority = priority if priority in ["low", "medium", "high"] else "medium"
        
        try:
            industry_templates = self.industry_templates[industry]
            category_templates = industry_templates.get(intent, industry_templates["default"])
            
            if isinstance(category_templates, dict):
                return category_templates.get(priority, category_templates["medium"])
            return category_templates
        except KeyError:
            return self.industry_templates["general"]["default"][priority]

    def _add_sentiment_phrases(self, response, sentiment):
        """Enhanced sentiment-aware phrasing"""
        if sentiment <= -0.7:
            return "[Urgent Priority] [Critical Issue] " + response
        elif sentiment <= -0.3:
            return "[Apologies for the inconvenience] " + response
        elif sentiment >= 0.7:
            return "[Thank you for your feedback] " + response
        return response

    def generate_response(self, ticket_data):
        """Improved response generation logic using embeddings"""
        intent = self._classify_intent_with_embedding(ticket_data.message)
        template = self._get_response_template(
            intent, ticket_data.industry, ticket_data.priority
        )
        
        response = template.replace("{priority}", ticket_data.priority.upper())
        response = self._add_sentiment_phrases(response, ticket_data.sentiment)
        
        closure = "\n\nBest regards,\n{company} Support"
        closures = {
            "tech": "\n\nTechnical Support Team",
            "retail": "\n\nCustomer Experience Team",
            "general": "\n\nCustomer Support Team"
        }
        response += closures.get(ticket_data.industry, closure)
        
        return response

# Initialize responder
responder = AutoResponder()

def generate_auto_response(message, industry, priority):
    """Improved Gradio interface wrapper"""
    ticket = TicketData(
        message=message,
        industry=industry,
        priority=priority
    )
    return responder.generate_response(ticket)

# Create enhanced Gradio interface
with gr.Blocks(title="Enhanced Auto-Responder", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 AI-Powered Customer Support Assistant")
    gr.Markdown("Generate context-aware responses for customer inquiries")
    
    with gr.Row():
        with gr.Column(scale=2):
            message_input = gr.Textbox(
                label="Customer Message",
                placeholder="Paste customer message here...",
                lines=5,
                max_lines=10
            )
            industry_input = gr.Dropdown(
                ["tech", "retail", "banking", "ecommerce", "general"],
                label="Industry Sector",
                value="tech"
            )
            priority_input = gr.Radio(
                ["low", "medium", "high"],
                label="Priority Level",
                value="medium"
            )
            submit_btn = gr.Button("Generate Response", variant="primary")
            
        with gr.Column(scale=3):
            output = gr.Textbox(
                label="Suggested Response",
                interactive=False,
                lines=10,
                show_copy_button=True
            )
    
    gr.Examples(
        examples=[
            ["URGENT: Can't access dashboard after update!", "tech", "high"],
            ["Package lost in transit order #12345", "retail", "medium"],
            ["When will my refund be processed?", "general", "low"]
        ],
        inputs=[message_input, industry_input, priority_input]
    )

    submit_btn.click(
        fn=generate_auto_response,
        inputs=[message_input, industry_input, priority_input],
        outputs=output,
        api_name="generate_response"
    )

if __name__ == "__main__":
    demo.launch()
