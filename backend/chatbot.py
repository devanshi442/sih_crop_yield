from googletrans import Translator
import random

translator = Translator()

# Predefined responses for various topics
RESPONSES = {
    "yield": [
        "Ensure proper irrigation and balanced fertilization to improve your crop yield.",
        "Monitor soil health and apply fertilizers as per crop requirements for better production.",
        "Regularly check for pests and diseases to avoid yield losses."
    ],
    "pest": [
        "Check your crops for signs of pest infestation and use recommended pesticides carefully.",
        "Consider natural pest control methods to protect your crops.",
        "Monitor pest alerts in your region and act early."
    ],
    "weather": [
        "Ensure irrigation if rainfall is low, and protect crops during high winds or storms.",
        "Check the local weather forecast to plan harvesting and irrigation.",
        "Extreme temperatures can affect your crops, so take protective measures."
    ],
    "fertilizer": [
        "Use fertilizers based on soil tests to avoid over or under fertilization.",
        "Organic compost can improve soil health along with chemical fertilizers.",
        "Follow crop-specific fertilizer schedules for best results."
    ],
    "msp": [
        "You can check the Minimum Support Price (MSP) for your crop from official sources.",
        "Selling at MSP ensures a fair price for your harvest."
    ],
    "default": [
        "I am here to help with your crops and farming queries.",
        "Please provide details about your crop or farm for advice.",
        "I can help with yield, pests, fertilizer, weather, and more."
    ]
}

# Keywords mapping (both English and Hindi)
KEYWORDS = {
    "yield": ["yield", "उपज", "फसल", "production", "area"],
    "pest": ["pest", "कीट", "insect", "infestation"],
    "weather": ["weather", "मौसम", "rain", "temperature", "बारिश", "तापमान"],
    "fertilizer": ["fertilizer", "उर्वरक", "fertilisation", "खाद"],
    "msp": ["msp", "न्यूनतम समर्थन मूल्य", "price"]
}

def get_chat_response(message: str) -> str:
    """
    Personalized multilingual chatbot for farming queries.
    """
    if not message:
        return "कृपया संदेश भेजें। / Please send a message."

    try:
        # Detect original language
        lang = translator.detect(message).lang

        # Translate to English for processing
        translated = translator.translate(message, dest="en").text.lower()

        # Match keywords to select topic
        topic_found = None
        for topic, keywords in KEYWORDS.items():
            if any(k.lower() in translated for k in keywords):
                topic_found = topic
                break

        # Choose a response
        if topic_found:
            reply_en = random.choice(RESPONSES[topic_found])
        else:
            reply_en = random.choice(RESPONSES["default"])

        # Translate back to original language if not English
        if lang != "en":
            reply = translator.translate(reply_en, dest=lang).text
        else:
            reply = reply_en

    except Exception as e:
        # fallback
        reply = f"You said: {message}"

    return reply
