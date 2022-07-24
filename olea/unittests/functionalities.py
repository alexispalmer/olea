# This contains tests to ensure that functionalities of 
# different aspects of the library are working as expected.

import unittest
from olea.utils.preprocess_text import PreprocessText as pt

class TestTextPreprocessing(unittest.TestCase):
    def test_text_preprocessing(self):
        example_text = [
            "THIS SHOULD BE LOWERCASED",
            "RT @amazing_user: Wow, what an amazing tweet",
            "https://fakelink.io http://anotherfakelink.io have you seen this? @another_user",
            "www.fakelink.io looks like this link doesn't start with http",
            "  WAY  too  many  spaces  in   this  message   ",
            "I can't believe such a thing is possible #amazing #themoreyouknow",
            "Incredible “thought” you have there", 
            "I prefer using ‘single’ quotes",
            "Wow I sure do love using ellipses … … …",
            "&#60;3",
            "I just LOVE the way you made this meal! 😍😍😍",
            "@user1 @user2 @user3 @user4 @user5 @user6 @user7 hello :)"
        ]

        expected_output = [
            "this should be lowercased",
            "USER wow, what an amazing tweet",
            "HTML HTML have you seen this? USER",
            "HTML looks like this link doesn't start with http"
            "way too many spaces in this message",
            "i can't believe such a thing is possible amazing the more you know",
            "incredible \"thought\" you have there",
            "i prefer using 'single' quotes",
            "wow i sure do love using ellipses ... ... ...",
            "<3",
            "i just love the way you made this meal! smiling face with heart-eyes smiling face with heart-eyes smiling face with heart-eyes",
            "USER USER USER hello :)"
        ]

        self.assertEqual(expected_output, pt.execute(example_text))

if __name__ == '__main__': 
    unittest.main()