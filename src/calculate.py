import math

def cal():
    
    action = int(input('Welcome to the simple math helper. \nWhat would you like to calculate? \n1. Sqrt \n2. Natural Log \n3. Factorial \n'))
    
    def factorial_recursive(n):
        if n == 1 or n == 0:
            return 1
        return n * factorial_recursive(n-1)

    if action == 1:
        number = int(input("Enter the number to get sqrt:"))
        output = math.sqrt(number)

    elif action == 2:
        number = int(input("Enter the number to get natural log:"))
        output = math.log(number)

    elif action == 3:
        number = int(input("Enter the number to get factorial:"))
        output = factorial_recursive(number)

    else:
        print("Entered invalid action.")
    return output

output = cal()
print("The answer is",output)

1