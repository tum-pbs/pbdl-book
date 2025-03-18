# source this file with "." in a shell

# do clean git checkout for changes from json-cleanup-for-pdf.py via:
# git checkout diffphys-code-burgers.ipynb diffphys-code-ns.ipynb diffphys-code-sol.ipynb physicalloss-code.ipynb bayesian-code.ipynb supervised-airfoils.ipynb reinflearn-code.ipynb physgrad-code.ipynb physgrad-comparison.ipynb physgrad-hig-code.ipynb


echo
echo WARNING - still requires one manual quit of first pdf/latex pass, use shift-x to quit, then fix latex
echo

PYT=python3

# warning - modifies notebooks!
${PYT} json-cleanup-for-pdf.py

# clean / remove _build dir ?

/Users/thuerey/Library/Python/3.9/bin/jupyter-book build .  --builder pdflatex

# manual?
#xelatex book

# unused fixup-latex.py


