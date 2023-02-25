import csv

def opener(filename):
    with open(filename) as csv_file:
        next(csv_file)
        for row in csv_file:
            fields = row.split(",")
            print("Alcohol: " + fields[0] + ", Mixer: " + fields[1] + "Food Pairings: " + fields[2])

def main():
    opener("csvReader/TastyDrinks.csv")
    
if __name__ == "__main__":
    main()