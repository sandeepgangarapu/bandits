#Examples
# "My name is Sandeep" - "Sandeep is name my"
# "a a" - "a a"
# "" - ""

# Code
def reverse_sentence(sen):
    new_sen = ''
    ln = len(sen)
    i = ln
    j = ln
    while(i > 0):
        while(sen[i] != ' '):
            i = i-1
        word = sen[i:j]
        new_sen += word
        i = i-1
        j = i
    return new_sen

print(reverse_sentence("My name is sandeep"))