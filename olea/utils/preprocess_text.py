# A helper function to preprocess text data

import re
import html
from emoji import demojize
import wordsegment as ws

class PreprocessText:
    messages  = []

    # Preprocessing function
    @classmethod
    def execute(cls, messages: list) -> list:
        """Function to preprocess the messages in COLD according to the
        paper.

        Args:
            messages (list): The column of messages to preprocess.  Must be
            passed as a list.

        Returns:
            list: The preprocessed messages as a list. 
        """
        # This is a pipelined process
        # 1. Lowercase all text, clean unicode characters
        # 2. Replace links with HTML and usernames with USER
        # 3. Segment hashtags into component words
        # 4. Replace emojis with word descriptions
        # 5. Limit consecutive USER mentions to 3 and final cleanup.

        cls.messages = messages

        # 1. Lowercase all text & clearn unicode characters
        preprocess = [x.lower() for x in cls.messages]
        preprocess = [html.unescape(x) for x in preprocess]
        # Replace some characters that can cause issues in other data formats
        # These are typically HTML codes that map to a specific character
        
        # Left double quote to regular double quote:
        preprocess = [x.replace("&#8220;", "\"") for x in preprocess]
        preprocess = [x.replace("“", "\"") for x in preprocess]
        # Right double quote to regular double quote:
        preprocess = [x.replace("&#8221;", "\"") for x in preprocess]
        preprocess = [x.replace("”", "\"") for x in preprocess]
        # Left single quote to regular single quote:
        preprocess = [x.replace("&#8216;", "\'") for x in preprocess]
        preprocess = [x.replace("‘", "\'") for x in preprocess]
        # Right single quote to regular single quote:
        preprocess = [x.replace("&#8217;", "\'") for x in preprocess]
        preprocess = [x.replace("â€¦", "\'") for x in preprocess]
        preprocess = [x.replace("â€™", "\'") for x in preprocess]
        preprocess = [x.replace("’", "\'") for x in preprocess]
        # Ellipses … to triple dot:
        preprocess = [x.replace("&#8230;", "...") for x in preprocess]
        preprocess = [x.replace("…", "...") for x in preprocess]

        # 2. Replace links with HTML and usernames with USER
        # Regex is used to find URLs and USER handles
        preprocess = [re.sub(r"\S*https?:\S*", " HTML ", x) 
                      for x in preprocess]
        preprocess = [re.sub(r"(rt )?@\S*", " USER ", x) for x in preprocess]

        # 3. Segment hashtags into component words
        preprocess = cls.__clean_hashtags(preprocess)

        # 4. Replace emojis with word descriptions
        preprocess = [demojize(x, delimiters = (" ", " ")) for x in preprocess]


        # 5. Limit consecutive USER mentions to 3
        # Trim redundant spaces
        preprocess = [re.sub(' {2,}', ' ', x) for x in preprocess]
        preprocess = [re.sub(r'^\s+', "", x) for x in preprocess]

        preprocess = [re.sub(r'(USER ){4,}', 
                        'USER USER USER ', x) 
                        for x in preprocess]

        # Some messages have </s><s> in them.  Removing them
        # Note from AP, DS: This might represent sarcasm in text.  
        preprocess = [x.replace("</s><s>", " ") for x in preprocess]
        preprocess = [x.replace("</s> <s>", " ") for x in preprocess]

        # Replace _ with a space so that each emoji is converted to its textual
        # tokens. (This is done here to prevent usernames with underscores from
        # earlier in the preprocessing from being wrongly replaced)
        preprocess = [x.replace("_", " ") for x in preprocess]

        # Strip trailing whitespace
        preprocess = [x.strip() for x in preprocess]

        return preprocess

    @classmethod
    def __clean_hashtags(cls, messages: list) -> list:
        """Private helper function to preprocess the messages in COLD 
        which contain a hashtag.

        Args:
            messages (list): The column of messages to preprocess.  Must be
            passed as a list.

        Returns:
            list: The preprocessed messages as a list. 
        """
        
        # The following is adapted from 
        # https://stackoverflow.com/questions/63822966/how-to-use-segment-from-wordsegment-inside-to-re-sub-to-extract-words-from-has
        ws.load()
        hashtag_processed_messages = []
        
        for message in range(len(messages)):
            message_string = messages[message]
            hashtags = re.findall(r"(#\w+)", message_string)

            # If there are hashtags in the message:
            if len(hashtags) != 0: 
                for hashtag in range(len(hashtags)):
                    identified_hashtag = hashtags[hashtag]
                    hashtag_words = " ".join(ws.segment(identified_hashtag))
                    message_string = message_string.replace(identified_hashtag, hashtag_words)
                hashtag_processed_messages.append(message_string)

            # If there are no hashtags in the message, append the message:
            else: 
                hashtag_processed_messages.append(message_string)

        return hashtag_processed_messages