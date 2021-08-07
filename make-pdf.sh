# source this file with "." in a shell

echo
echo WARNING - still requires one manual quit of first pdf/latex pass, use shift-x to quit
echo

# do clean git checkout for changes from json-cleanup-for-pdf.py?
# git checkout diffphys-code-burgers.ipynb diffphys-code-sol.ipynb physicalloss-code.ipynb bayesian-code.ipynb supervised-airfoils.ipynb

# warning - modifies notebooks!
python3.7 json-cleanup-for-pdf.py

# clean / remove _build dir ?

# GEN!
/Users/thuerey/Library/Python/3.7/bin/jupyter-book build . --builder pdflatex

cd _build/latex
#mv book.pdf book-xetex.pdf # failed anyway

rm -f book-in.tex sphinxmessages-in.sty book-in.aux book-in.toc
mv book.tex book-in.tex
mv sphinxmessages.sty sphinxmessages-in.sty
mv book.aux book-in.aux
mv book.toc book-in.toc
mv sphinxmanual.cls sphinxmanual-in.cls

python3.7 ../../fixup-latex.py
# generates book-in2.tex

# remove unicode chars
iconv -c -f utf-8 -t ascii book-in2.tex > book.tex

# finally run pdflatex, now it should work:
# pdflatex -recorder book
pdflatex book
pdflatex book

mv book.pdf ../../book-pdflatex.pdf

tar czvf ../../pbdl-latex-for-arxiv.tar.gz *


