# Ex.No: 9  Logic Programming –  Computer Maintenance Expert System
### DATE:  23-09-24                                                                         
### REGISTER NUMBER : 212222040152
### AIM: 
Write a Prolog program to build a computer maintenance expert system.
###  Algorithm:
1. Start the program.
2. Write the rules for each fault in computer.
3. If system have printing problem, missing dots and no uniform printing then system fault on printer head.
4. If system have not printing, missing dots and spread inks then system fault on ribbon
5. If system have not printing, paper jam and out of paper then system fault on paper stuck in printer
6. Similarly define rules for all faults.
7. Define facts for system problems.
8. Find the fault of computer by passing query to system.
     
### Program:

```py
% Define facts
symptom(printer, no_printing).
symptom(printer, missing_dots).
symptom(printer, no_uniform_printing).
symptom(printer, spread_ink).
symptom(printer, paper_jam).
symptom(printer, out_of_paper).

% Define rules for diagnosis
fault(printer_head) :- 
    symptom(printer, no_printing), 
    symptom(printer, missing_dots), 
    symptom(printer, no_uniform_printing).

fault(ribbon) :- 
    symptom(printer, no_printing), 
    symptom(printer, missing_dots), 
    symptom(printer, spread_ink).

fault(paper_stuck) :- 
    symptom(printer, no_printing), 
    symptom(printer, paper_jam), 
    symptom(printer, out_of_paper).

% Query examples
find_fault(Fault) :- fault(Fault).

```

### Output:
![image](https://github.com/user-attachments/assets/fcb0ff0f-b112-45f8-adf0-3683e77cc644)

### Result:
Thus the simple omputer maintenance expert system was built sucessfully.
