from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
from IPython.display import YouTubeVideo 
import whisper
from pytube import YouTube
from langdetect import detect
from googletrans import Translator
from transformers import BartForConditionalGeneration, BartTokenizer
import spacy
from IPython.display import HTML


#videourl = "https://youtu.be/GI3oYDGtxZA?si=0smmK-IqfoAXFxfC"

# Extract video ID from the URL
#video_id = videourl.split("/")[-1]




#video = YouTubeVideo(video_id)



def transcribe(url):
    wspr = whisper.load_model('base')
    ytrl = YouTube(url)
    streams = ytrl.streams.filter(only_audio = True)
    stream = streams.first()
    stream.download(filename = 'audio1.mp4')
    transcript = wspr.transcribe('audio1.mp4')
    transcript = transcript['text']
    return transcript

'''
try:
    
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    all_languages = ["hi", "bn", "te", "ta", "mr", "en", "es", "fr", "de", "zh-Hans",'ar']


    total_content = ''
    for transcript in transcript_list:
        if  transcript.language_code in all_languages:
            captions = transcript.fetch()
            for segment in captions:
                if segment['text'] != '':  # Skip empty segments
                    total_content += segment['text'] + ' '

    print(total_content)
except:
    print("No transcript available...so creating one..")
    print('please wait for few minutes..')
    total_content = transcribe(videourl)
    print(total_content)'''
    


def detect_language(text):
    try:
        language = detect(text)
        return language
    except Exception as e:
        print(f"Error detecting language: {e}")
        return None

# Example usage
#text_to_detect = "यह तोह बोहोत ही आसान है "

'''detected_language = detect_language(total_content)

if detected_language:
    print(f"The detected language is: {detected_language}")
else:
    print("Language detection failed.")'''



def translate_large_text(text, src_language, dest_language='en', chunk_size=500):
    translator = Translator()

    # Split the text into chunks
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    # Translate each chunk
    translated_chunks = []
    for chunk in chunks:
        try:
            translated_chunk = translator.translate(chunk, src=src_language, dest=dest_language)
            translated_chunks.append(translated_chunk.text)
        except Exception as e:
            print(f"Translation error: {e}")
    print('enter')
    # Combine translated chunks into the final result
    translated_text = ' '.join(translated_chunks)

    return translated_text

# Example usage
'''if detected_language != 'en':

    total_content = translate_large_text(total_content, src_language=detected_language)
    print(total_content)

len(total_content.split(' '))'''


# Load the model and tokenizer
model = BartForConditionalGeneration.from_pretrained(
    'facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained(
    'facebook/bart-large-cnn')


all_summeries = []

def summarize(text, maxSummarylength=500):
    # Encode the text and summarize
    inputs = tokenizer.encode("summarize: " +
                              text,
                              return_tensors="pt",
                              max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=int(maxSummarylength/3*2),
                                 min_length=int(maxSummarylength/5),
                                 length_penalty=10.0,
                                 num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
def split_text_into_pieces(text,
                           max_tokens=900,
                           overlapPercent=10):
    # Tokenize the text
    tokens = tokenizer.tokenize(text)

    # Calculate the overlap in tokens
    overlap_tokens = int(max_tokens * overlapPercent / 100)

    # Split the tokens into chunks of size
    # max_tokens with overlap
    pieces = [tokens[i:i + max_tokens]
              for i in range(0, len(tokens),
                             max_tokens - overlap_tokens)]

    print('pieces======>>>>>>>>>>>',pieces)

    # Convert the token pieces back into text
    text_pieces = [tokenizer.decode(
        tokenizer.convert_tokens_to_ids(piece),
        skip_special_tokens=True) for piece in pieces]

    return text_pieces


def recursive_summarize(text, max_length=150, recursionLevel=0):
    recursionLevel=recursionLevel+1
    print("######### Recursion level: ",
          recursionLevel,"\n\n######### ")
    tokens = tokenizer.tokenize(text)
    expectedCountOfChunks = len(tokens)/max_length
    max_length=int(len(tokens)/expectedCountOfChunks)+2
   
    print(max_length)
    # Break the text into pieces of max_length
    pieces = split_text_into_pieces(text, max_tokens=max_length)

    print("Number of pieces: ", len(pieces))
    # Summarize each piece
    summaries=[]
    k=0
    for k in range(0, len(pieces)):
        piece=pieces[k]
        #print("****************************************************")
        #print("Piece:",(k+1)," out of ", len(pieces), "pieces")
        #print(piece, "\n")
        summary =summarize(piece, maxSummarylength=max_length)
        #print("SUMNMARY: ", summary)
        summaries.append(summary)
        #print("****************************************************")

    concatenated_summary = ' '.join(summaries)
    all_summeries.append(concatenated_summary)
    print('concat_summary',concatenated_summary)

    tokens = tokenizer.tokenize(concatenated_summary)
    print('length==',len(tokens))
    if len(tokens) > max_length:
        # If the concatenated_summary is too long, repeat the process
        print("############# GOING RECURSIVE ##############")
        return recursive_summarize(concatenated_summary,
                                   max_length=max_length,
                                   recursionLevel=recursionLevel)
    else:
      # Concatenate the summaries and summarize again
        final_summary=concatenated_summary
        '''if len(pieces)>1:
            final_summary = summarize(concatenated_summary,
                                  maxSummarylength=max_length)'''
        return all_summeries
# Example usage

#final_summary = recursive_summarize(total_content)

#print(all_summeries)
#print("\n%%%%%%%%%%%%%%%%%%%%%\n")
#print("Final summary:", final_summary)


def sentence_importance(sentence):
    return sum([word.vector_norm for word in sentence])

def summarize_paragraph(paragraph, num_sentences=3):
    nlp = spacy.load("en_core_web_sm")
    
    # Process the paragraph with spaCy
    doc = nlp(paragraph)
    
    # Get sentences and their respective importance scores
    sentences = [(sent, sentence_importance(sent)) for sent in doc.sents]
    
    # Sort sentences by importance score
    sorted_sentences = sorted(sentences, key=lambda x: x[1], reverse=True)
    
    # Get the top N sentences
    top_sentences = sorted_sentences[:num_sentences]
    
    # Sort top sentences by their original order in the paragraph
    top_sentences = sorted(top_sentences, key=lambda x: x[0].start)
    
    # Extract the text of the top sentences
    summarized_text = [sent.text.strip() for sent, _ in top_sentences]
    
    return summarized_text

# Example usage
paragraph = """
The company's financial performance in the last quarter has been outstanding. 
Revenues exceeded expectations, showing a 15% growth compared to the previous year. 
The key drivers of this success were innovative product launches and effective cost management strategies.
"""

#result = summarize_paragraph(all_summeries[-1], num_sentences=5)

# Print the summarized text
#for i, sentence in enumerate(result, 1):
    #print(f"{i}. {sentence}")



def identify_key_phrases(text, min_length=2, max_length=4):
    try:
        # Load the English NLP model from spaCy
        nlp = spacy.load("en_core_web_sm")
        
        # Process the input text
        doc = nlp(text)
        
        # Extract key phrases (noun chunks) of specific lengths from the processed document
        key_phrases = [chunk.text for chunk in doc.noun_chunks if min_length <= len(chunk) <= max_length]
        
        return key_phrases
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage
text = "Natural Language Processing is a field of study that focuses on the interaction between computers and humans using natural language."
#key_phrases = identify_key_phrases(all_summeries[-1], min_length=2, max_length=4)

#if key_phrases is not None:
    #print("Key Phrases (lengths 2, 3, and 4):")
    #print(key_phrases)


def highlight_key_phrases(notes, key_phrases):
    highlighted_notes = notes
    for phrase in key_phrases:
        highlighted_notes = highlighted_notes.replace(phrase, f'<span style="color:red; font-weight:bold;">{phrase}</span>')
    return highlighted_notes

#highlighted_notes = highlight_key_phrases(''.join(result), key_phrases)

# Display the highlighted notes as HTML
#for i, sentence in enumerate(highlighted_notes.split('.'), 1):
    #print(f"{i}. {sentence}")