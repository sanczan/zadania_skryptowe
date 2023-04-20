101.
def factorial(n):
    x = 1
    for i in range(1, n + 1):
        x *= i
    return x

print(factorial(5))

102.
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

print(factorial(5))

103.
import re

url = input("Podaj adres url: ")

pattern = re.compile("^(?:http(s)?:\/\/)?[\w.-]+(?:\.[\w\.-]+)+[\w\-\._~:/?#[\]@!\$&'\(\)\*\+,;=.]+$")

if pattern.match(url):
    print("Ten adres jest poprawny")
else:
    print("Ten adres nie jest poprawny.")

104. 
def bubbleSort(input):
    new = input

    for i in range(0, len(new) - 1):
        for j in range(0, len(new) - 1):
            if new[j] > new[j + 1]:
                new[j], new[j+1] = new[j+1], new[j]

    return new

def insertionSort(input):
    new = input

    for j in range(2, len(new)):
        key = new[j]
        i = j-1
        while i >= 0 and new[i] > key:
            new[i+1] = new[i]
            i -= 1
        new[i+1] = key

    return  new

def quickSort(input):

    def partition(input, p, k):
        x = input[k]
        i = p - 1

        for j in range(p, k):
            if input[j] <= x:
                i += 1
                input[i], input[j] = input[j], input[i]

        input[k], input[i + 1] = input[i + 1], input[k]

        return i + 1

    def main(input, p, k):
        if p < k:
            q = partition(input, p, k)
            main(input, p, q - 1)
            main(input, q + 1, k)

    new = input
    main(new, 0, len(new) - 1)

    return new


numbers = []

try:

    file = open("danezadanie104.txt", "r")

    for line in file.readlines():
        try:
            numbers.append(int(line))
        except ValueError:
            continue

    file.close()

    ## BUBBLE SORT

    bubbleFile = open("zadanie104_bubble.txt", "w")

    for number in bubbleSort(numbers):
        bubbleFile.write(str(number) + "\n")

    bubbleFile.close()


    ## INSERTION SORT

    insertionFile = open("zadanie104_insertion.txt", "w")

    for number in insertionSort(numbers):
        insertionFile.write(str(number) + "\n")

    insertionFile.close()


    ## QUICK SORT

    quickFile = open("zadanie104_quick.txt", "w")

    for number in quickSort(numbers):
        quickFile.write(str(number) + "\n")

    quickFile.close()

except IOError:
    print("Blad otwierania pliku")

105.
def averageMinMax(input):
    return (max(input)+min(input))/2

print(averageMinMax([1,2,3,5,-3,0,9.7]))

106.
import random
import time
import datetime

def bubbleSort(input):
    new = input

    for i in range(0, len(new) - 1):
        for j in range(0, len(new) - 1):
            if new[j] > new[j + 1]:
                new[j], new[j+1] = new[j+1], new[j]

    return new

def insertionSort(input):
    new = input

    for j in range(2, len(new)):
        key = new[j]
        i = j-1
        while i >= 0 and new[i] > key:
            new[i+1] = new[i]
            i -= 1
        new[i+1] = key

    return  new

def quickSort(input):

    def partition(input, p, k):
        x = input[k]
        i = p - 1

        for j in range(p, k):
            if input[j] <= x:
                i += 1
                input[i], input[j] = input[j], input[i]

        input[k], input[i + 1] = input[i + 1], input[k]

        return i + 1

    def main(input, p, k):
        if p < k:
            q = partition(input, p, k)
            main(input, p, q - 1)
            main(input, q + 1, k)

    new = input
    main(new, 0, len(new) - 1)

    return new


numbers = []

for i in range(900):
    numbers.append(random.randint(1, 100))

def writeToFile(i, name, count, time):
    file.write(str(i) + ". " + str(name).ljust(20) + str(count).ljust(30) + str(str(time) + " ms").ljust(30) + "\n")

try:

    file = open("raport_" + str(datetime.date.today()) +  ".txt", "w")

    file.write("RAPORT ALGORYTMOW SORTOWANIA\n\n")
    file.write("NAZWA".ljust(23) + "LICZBA SORTOWANYCH PLIKOW".ljust(30) + "SREDNI CZAS".ljust(30) + "\n")

    startTime = time.time()
    sorted = bubbleSort(numbers)
    stopTime = time.time()
    result = (stopTime - startTime) * 1000
    writeToFile(1, "bąbelkowe", len(numbers), result)

    startTime = time.time()
    sorted = insertionSort(numbers)
    stopTime = time.time()
    result = (stopTime - startTime) * 1000
    writeToFile(2, "wstawianie", len(numbers), result)

    startTime = time.time()
    sorted = quickSort(numbers)
    stopTime = time.time()
    result = (stopTime - startTime) * 1000
    writeToFile(3, "quick sort", len(numbers), result)

    file.close()

except IOError:
    print("Blad otwierania pliku")

107.
def encrypt(inputText, shift):

    output = ""

    for ch in inputText.lower():

        if ch.isalpha():
            shiftedChar = ord(ch) + shift
        else:
            output += ' '
            continue

        if shiftedChar > ord('z'):
            shiftedChar -= 26

        output += chr(shiftedChar)

    return output


def decrypte(inputText, shift):

    output = ""

    for c in inputText.lower():

        if c.isalpha():
            shiftedChar = ord(c) - shift
        else:
            output += ' '
            continue

        if shiftedChar > ord('z'):
            shiftedChar += 26

        output += chr(shiftedChar)

    return output


print(encrypt("lubie placki", 3))
print(decrypte("dod pd nrwd", 3))

108.
def assign(nodes, label, result, prefix = ''):    
    childs = nodes[label]     
    tree = {}
    if len(childs) == 2:
        tree['0'] = assign(nodes, childs[0], result, prefix+'0')
        tree['1'] = assign(nodes, childs[1], result, prefix+'1')     
        return tree
    else:
        result[label] = prefix
        return label

def huffman(_vals):    
    vals = _vals.copy()
    nodes = {}
    for n in vals.keys():
        nodes[n] = []

    while len(vals) > 1:
        s_vals = sorted(vals.items(), key=lambda x: x[1])
        a1 = s_vals[0][0]
        a2 = s_vals[1][0]
        vals[a1+a2] = vals.pop(a1) + vals.pop(a2)
        nodes[a1+a2] = [a1, a2]        
    code = {}
    root = a1+a2
    tree = {}
    tree = assign(nodes, root, code)
    return code, tree

freq = [(8.167, 'a'), (1.492, 'b'), (2.782, 'c'), (4.253, 'd'),
        (12.702, 'e'),(2.228, 'f'), (2.015, 'g'), (6.094, 'h'),
        (6.966, 'i'), (0.153, 'j'), (0.747, 'k'), (4.025, 'l'),
        (2.406, 'm'), (6.749, 'n'), (7.507, 'o'), (1.929, 'p'),
        (0.095, 'q'), (5.987, 'r'), (6.327, 's'), (9.056, 't'),
        (2.758, 'u'), (1.037, 'v'), (2.365, 'w'), (0.150, 'x'),
        (1.974, 'y'), (0.074, 'z')]

############################

vals = {l: v for (v, l) in freq}
code, tree = huffman(vals)

text = 'python'
encoded = ''.join([code[t] for t in text])
print('Zakodowany tekst:', encoded)

decoded = []
i = 0
while i < len(encoded):
    ch = encoded[i]
    act = tree[ch]
    while not isinstance(act, str):
        i += 1
        ch = encoded[i]
        act = act[ch]
    decoded.append(act)
    i += 1

print('Odkowowany tekst:', ''.join(decoded))

109.
def assign(nodes, label, result, prefix = ''):
    childs = nodes[label]
    tree = {}
    if len(childs) == 2:
        tree['0'] = assign(nodes, childs[0], result, prefix+'0')
        tree['1'] = assign(nodes, childs[1], result, prefix+'1')
        return tree
    else:
        result[label] = prefix
        return label

def huffman(_vals):
    vals = _vals.copy()
    nodes = {}
    for n in vals.keys():
        nodes[n] = []

    while len(vals) > 1:
        s_vals = sorted(vals.items(), key=lambda x: x[1])
        a1 = s_vals[0][0]
        a2 = s_vals[1][0]
        vals[a1+a2] = vals.pop(a1) + vals.pop(a2)
        nodes[a1+a2] = [a1, a2]
    code = {}
    root = a1+a2
    tree = {}
    tree = assign(nodes, root, code)
    return code, tree

freq = [(8.167, 'a'), (1.492, 'b'), (2.782, 'c'), (4.253, 'd'),
        (12.702, 'e'),(2.228, 'f'), (2.015, 'g'), (6.094, 'h'),
        (6.966, 'i'), (0.153, 'j'), (0.747, 'k'), (4.025, 'l'),
        (2.406, 'm'), (6.749, 'n'), (7.507, 'o'), (1.929, 'p'),
        (0.095, 'q'), (5.987, 'r'), (6.327, 's'), (9.056, 't'),
        (2.758, 'u'), (1.037, 'v'), (2.365, 'w'), (0.150, 'x'),
        (1.974, 'y'), (0.074, 'z')]

############################

vals = {l: v for (v, l) in freq}
code, tree = huffman(vals)

text = 'python'
encoded = ''.join([code[t] for t in text])
print('Zakodowany tekst:', encoded)

decoded = []
i = 0
while i < len(encoded):
    ch = encoded[i]
    act = tree[ch]
    while not isinstance(act, str):
        i += 1
        ch = encoded[i]
        act = act[ch]
    decoded.append(act)
    i += 1

print('Odkowowany tekst:', ''.join(decoded))

110.
def gauss(a, b):
    n = len(a)
    p = len(b[0])
    det = 1
    for i in range(n - 1):
        k = i
        for j in range(i + 1, n):
            if abs(a[j][i]) > abs(a[k][i]):
                k = j
        if k != i:
            a[i], a[k] = a[k], a[i]
            b[i], b[k] = b[k], b[i]
            det = -det

        for j in range(i + 1, n):
            t = a[j][i] / a[i][i]
            for k in range(i + 1, n):
                a[j][k] -= t * a[i][k]
            for k in range(p):
                b[j][k] -= t * b[i][k]

    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            t = a[i][j]
            for k in range(p):
                b[i][k] -= t * b[j][k]
        t = 1 / a[i][i]
        det *= a[i][i]
        for j in range(p):
            b[i][j] *= t
    return b

def printMatrix(A):
    n = len(A)
    for i in range(n):
        line = "|"
        for j in range(n):
            line += str(A[i][j]).center(25)
            if j == n-1:
                line += "|"
        print(line)
    print("")


a = [[2, 9, 4], [7, 5, 3], [6, 1, 8]]
b = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
c = gauss(a, b)

printMatrix(c)

111.
import datetime

date = datetime.datetime.strptime('Dec 3 2018  1:33PM', '%b %d %Y %I:%M%p')

print(date)

112.
import datetime
print(datetime.datetime.now())

113.
import datetime

inputData = input("in: ")
inputComponents = inputData.split(",")

date = datetime.date(int(inputComponents[0]), 1, 1)
week = int(inputComponents[1])

while(date.isocalendar()[1] < week):
    date += datetime.timedelta(days=1)

while(date.weekday() > 0):
    date += datetime.timedelta(days=1)


print("out: " + str(date.ctime()))

114.
import math

numbers = input("in: ")

n = float(numbers.split(",")[0])
a = float(numbers.split(",")[1])

result = n/4 * 1/(math.tan(math.pi/n)) * a**2

print("out: " + str(result))

115.
import math

class ComplexNumber(object):

    def __init__(self, real, imag=0.0):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        return ComplexNumber(self.real + other.real,
                       self.imag + other.imag)

    def __sub__(self, other):
        return ComplexNumber(self.real - other.real,
                       self.imag - other.imag)

    def __mul__(self, other):
        return ComplexNumber(self.real*other.real - self.imag*other.imag,
                       self.imag*other.real + self.real*other.imag)

    def __truediv__(self, other):
        r = float(other.real**2 + other.imag**2)
        return ComplexNumber((self.real*other.real+self.imag*other.imag)/r, (self.imag*other.imag-self.real*other.imag)/r)

    def __abs__(self):
        return math.sqrt(self.real**2 + self.imag**2)

    def __str__(self):
        if self.imag >= 0:
            return '(%g+%gi)' % (self.real, self.imag)
        return '(%g%gi)' % (self.real, self.imag)

print((-23+0j) + (17+9j))
print(ComplexNumber(-23, 0) + ComplexNumber(17, 9))

116.
import math

class ComplexNumber(object):

    def __init__(self, real, imag=0.0):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        return ComplexNumber(self.real + other.real,
                       self.imag + other.imag)

    def __sub__(self, other):
        return ComplexNumber(self.real - other.real,
                       self.imag - other.imag)

    def __mul__(self, other):
        return ComplexNumber(self.real*other.real - self.imag*other.imag,
                       self.imag*other.real + self.real*other.imag)

    def __truediv__(self, other):
        r = float(other.real**2 + other.imag**2)
        return ComplexNumber((self.real*other.real+self.imag*other.imag)/r, (self.imag*other.imag-self.real*other.imag)/r)

    def __abs__(self):
        return math.sqrt(self.real**2 + self.imag**2)

    def __str__(self):
        if self.imag >= 0:
            return '(%g+%gi)' % (self.real, self.imag)
        return '(%g%gi)' % (self.real, self.imag)

print((-23+0j) * (17+9j))
print(ComplexNumber(-23, 0) * ComplexNumber(17, 9))

117.
import math

class ComplexNumber(object):

    def __init__(self, real, imag=0.0):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        return ComplexNumber(self.real + other.real,
                       self.imag + other.imag)

    def __sub__(self, other):
        return ComplexNumber(self.real - other.real,
                       self.imag - other.imag)

    def __mul__(self, other):
        return ComplexNumber(self.real*other.real - self.imag*other.imag,
                       self.imag*other.real + self.real*other.imag)

    def __truediv__(self, other):
        r = float(other.real**2 + other.imag**2)
        return ComplexNumber((self.real*other.real+self.imag*other.imag)/r, (self.imag*other.imag-self.real*other.imag)/r)

    def __abs__(self):
        return math.sqrt(self.real**2 + self.imag**2)

    def __str__(self):
        if self.imag >= 0:
            return '(%g+%gi)' % (self.real, self.imag)
        return '(%g%gi)' % (self.real, self.imag)

print((-23+0j) / (17+9j))
print(ComplexNumber(-23, 0) / ComplexNumber(17, 9))

118.
import math

class ComplexNumber(object):

    def __init__(self, real, imag=0.0):
        self.real = real
        self.imag = imag

    def __add__(self, other):
        return ComplexNumber(self.real + other.real,
                       self.imag + other.imag)

    def __sub__(self, other):
        return ComplexNumber(self.real - other.real,
                       self.imag - other.imag)

    def __mul__(self, other):
        return ComplexNumber(self.real*other.real - self.imag*other.imag,
                       self.imag*other.real + self.real*other.imag)

    def __truediv__(self, other):
        r = float(other.real**2 + other.imag**2)
        return ComplexNumber((self.real*other.real+self.imag*other.imag)/r, (self.imag*other.imag-self.real*other.imag)/r)

    def __abs__(self):
        return math.sqrt(self.real**2 + self.imag**2)

    def __str__(self):
        if self.imag >= 0:
            return '(%g+%gi)' % (self.real, self.imag)
        return '(%g%gi)' % (self.real, self.imag)

print(abs(-23-8j))
print(abs(ComplexNumber(-23, -8)))

119.
import cmath

number = input("in: ")
print("out: " + str(cmath.phase(complex(number))))

120.
from fractions import Fraction

def getNumber(prompt):

    while(True):

        value = input(prompt)

        try:
            n = float(value)
            return n

        except ValueError:
            print("Blad!")

print("out: " + str(Fraction(getNumber("in: ")).limit_denominator()))

121.
class randomGenerator(object):

    def __init__(self, seed=3):
        self.seed = seed

    def random(self):
        self.seed = (1103515245 * self.seed + 12345) & 0x7fffffff
        return self.seed

generator = randomGenerator(35)

print(generator.random())

122.
import random

input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

output = []

while(len(input)>0):
    index = random.randint(0, len(input)-1)
    output.append(input[index])
    del input[index]

print(output)

123.
import random

def randomItem(input):
    return input[random.randint(0, len(input)-1)]

input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(randomItem(input))


124.
romanMap = [(1000, 'M'),
            (900, 'CM'),
            (500, 'D'),
            (400, 'CD'),
            (100, 'C'),
            (90, 'XC'),
            (50, 'L'),
            (40, 'XL'),
            (10, 'X'),
            (9, 'IX'),
            (5, 'V'),
            (4, 'IV'),
            (1, 'I')]


def rom(input):

    output = ""

    while input > 0:
        for i, r in romanMap:
            while input >= i:
                output += r
                input -= i

    return output


print(rom(1706))

125.
romanMap = [(1, 'I'),
            (5, 'V'),
            (10, 'X'),
            (50, 'L'),
            (100, 'C'),
            (500, 'D'),
            (1000, 'M')]

def derom(input):

    output = 0

    for i in range(0, len(input)):

        for each in romanMap:

            if input[i] == each[1]:

                if each[0] > output:
                    output = each[0] - output
                else:
                    output += each[0]

    return output

print(derom("MDCCVI"))

126.
import math

class Kolo(object):

    def __init__(self, r):
        self.r = float(r)

    def pole(self):
        return math.pi*(self.r**2)

    def obwod(self):
        return 2*math.pi*self.r

kolo = Kolo(17)
print(kolo.pole())
print(kolo.obwod())

127.
class Reversor(object):

    def __init__(self, text):
        self.text = str(text)

    def __str__(self):
        output = ""
        words = self.text.split(' ')

        i = 0

        while(i < len(words)-1):
            output += (words[i+1] + " " + words[i] + " ")
            i+=2

        if len(words)%2 != 0:
            output += words[len(words)-1]

        return output

reversor = Reversor(input("in: "))
print("out: " + str(reversor))

128.
class Matma:
    staticmethod
    def pow(self, x, n):
        if x == 0 or x == 1 or n == 1:
            return x
        result = 1
        for _ in range(n):
            result *= x
        return result

matma = Matma()

print(matma.pow(2,8))

129.
def absolute(number):

    if number < 0:
        return -number

    return number

print(absolute(-17))

130.
string = input("Podaj zdanie: ")

count = 0

for letter in string.lower():
    if letter in "aeyioąęuó":
        count += 1

print(count)