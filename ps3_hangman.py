# 6.00 Problem Set 3
# 
# Hangman game
#

# -----------------------------------
# Helper code
# You don't need to understand this helper code,
# but you will have to know how to use the functions
# (so be sure to read the docstrings!)

import random
import string

WORDLIST_FILENAME = "words.txt"

def loadWords():
    """
    Returns a list of valid words. Words are strings of lowercase letters.
    
    Depending on the size of the word list, this function may
    take a while to finish.
    """
    print "Loading word list from file..."
    # inFile: file
    inFile = open(WORDLIST_FILENAME, 'r', 0)
    # line: string
    line = inFile.readline()
    # wordlist: list of strings
    wordlist = string.split(line)
    print "  ", len(wordlist), "words loaded."
    return wordlist

def chooseWord(wordlist):
    """
    wordlist (list): list of words (strings)

    Returns a word from wordlist at random
    """
    return random.choice(wordlist)

# end of helper code
# -----------------------------------

# Load the list of words into the variable wordlist
# so that it can be accessed from anywhere in the program
wordlist = loadWords()

def isWordGuessed(secretWord, lettersGuessed):
    '''
    secretWord: string, the word the user is guessing
    lettersGuessed: list, what letters have been guessed so far
    returns: boolean, True if all the letters of secretWord are in lettersGuessed;
      False otherwise
    '''
    ##guess = lettersGuessed.values()
    guess = []
    used = []
    i = 0
    guessed = True
    listSecret = list(secretWord)
    #listSecret.sort()
    
    while guessed:
        for char in secretWord:
            if char in lettersGuessed:
                guess.append(char)
        else:
            guessed = False
    
    if guess == listSecret:
        #print "Good guess:", secretWord
        return True
    else:
        return False



def getGuessedWord(secretWord, lettersGuessed):
    '''
    secretWord: string, the word the user is guessing
    lettersGuessed: list, what letters have been guessed so far
    returns: string, comprised of letters and underscores that represents
      what letters in secretWord have been guessed so far.
    '''
    #Determine length of random word and display number of blanks
    blanks = '_ ' * len(secretWord)
    #print ()
    #print ("Word: ",blanks)
    #letterIndex = []
    #guessed = True
    
    #while guessed:
        #for char in secretWord:
            #if char in lettersGuessed:
    newBlanks = "".join(c if c in lettersGuessed else "_" for c in secretWord)
    return newBlanks 



def getAvailableLetters(lettersGuessed):
    '''
    lettersGuessed: list, what letters have been guessed so far
    returns: string, comprised of letters that represents what letters have not
      yet been guessed.
    '''
    import string
    availableLetters = string.ascii_lowercase
    #print choices
    lettersGuessed = ''.join(sorted(lettersGuessed))
    i = 0
    #Guessed = True
    
    #if lettersGuessed not in availableLetters:
        #print "Oops! You've already guessed that letter: ", lettersGuessed 
     #   return not Guessed
    
    while i < len(lettersGuessed):
        for char in lettersGuessed:
            i += 1
            if char in availableLetters:
                availableLetters = availableLetters.replace(char, "")
                #return False
                
    return availableLetters        
    
    

def hangman(secretWord):
    '''
    secretWord: string, the secret word to guess.

    Starts up an interactive game of Hangman.

    * At the start of the game, let the user know how many 
      letters the secretWord contains.

    * Ask the user to supply one guess (i.e. letter) per round.

    * The user should receive feedback immediately after each guess 
      about whether their guess appears in the computers word.

    * After each round, you should also display to the user the 
      partially guessed word so far, as well as letters that the 
      user has not yet guessed.

    Follows the other limitations detailed in the problem write-up.
    '''
    print "Welcome to the game, Hangman!"
    #secretWord = chooseWord(wordlist)
    print "I am thinking of a word", len(secretWord), "letters long."

    lettersGuessed = ""
    mistakesMade = 0
    #lose = "Sorry, you ran out of guesses. The word was", secretWord
    while mistakesMade < 8:
        print "-----------\nYou have", 8-mistakesMade, "guesses left."
        
        print "Available letters: ", getAvailableLetters(lettersGuessed)
        #print 'guessed Letters', lettersGuessed
        guess =  raw_input("Please guess a letter: ")
               
        if guess in lettersGuessed:       
            print "Oops! You've already guessed that letter: ", \
            getGuessedWord(secretWord, lettersGuessed)
        elif guess not in lettersGuessed:
            lettersGuessed += guess
        
            if getAvailableLetters(lettersGuessed):
            #print "True"
                if guess in secretWord:
                    print "Good guess:", getGuessedWord(secretWord, \
                    lettersGuessed)
                if guess not in secretWord:
                    print "Oops! That letter is not in my word: ", \
                    getGuessedWord(secretWord, lettersGuessed)
                    mistakesMade += 1
                if isWordGuessed(secretWord, lettersGuessed):
                    print "-----------\nCongratulations, you won!"
                    break
    if mistakesMade == 8:
        print "-----------\nSorry, you ran out of guesses. The word was", \
        secretWord            
            
secretWord = chooseWord(wordlist)
hangman(secretWord)




# When you've completed your hangman function, uncomment these two lines
# and run this file to test! (hint: you might want to pick your own
# secretWord while you're testing)

# secretWord = chooseWord(wordlist).lower()
# hangman(secretWord)