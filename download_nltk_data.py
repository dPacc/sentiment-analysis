import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download specific packages
nltk.download('punkt')
nltk.download('punkt_tab')  # This is what the error is asking for
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Force download all popular packages 
nltk.download('all')

print("NLTK downloads completed successfully!")