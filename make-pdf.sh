# source this file with "." in a shell

echo
echo Note: manually quit first latex pass with shift-x 
echo

cd /Users/thuerey/Dropbox/mbaDevelSelected/pbdl-book/
/Users/thuerey/Library/Python/3.7/bin/jupyter-book build . --builder pdflatex

cd /Users/thuerey/Dropbox/mbaDevelSelected/pbdl-book/_build/latex
export JPYFILENAME=book.tex
rm ${JPYFILENAME}-in.bak
mv ${JPYFILENAME} ${JPYFILENAME}-in.bak
iconv -c -f utf-8 -t ascii ${JPYFILENAME}-in.bak > ${JPYFILENAME}

echo running SED
sed -i '' -e 's/sphinxincludegraphics{{/sphinxincludegraphics{/g' ${JPYFILENAME}
sed -i '' -e 's/}.png}/.png}/g' ${JPYFILENAME}
sed -i '' -e 's/}.jpg}/.jpg}/g' ${JPYFILENAME}
sed -i '' -e 's/}.jpeg}/.jpeg}/g' ${JPYFILENAME}
sed -i '' -e 's/sphinxpxdimen]{{/sphinxpxdimen]{/g' ${JPYFILENAME}

# fix chapters
# sed -i '' -e 's///g' ${JPYFILENAME}
# sed -i '' -e 's///g' ${JPYFILENAME}
# sed -i '' -e 's///g' ${JPYFILENAME}
sed -i '' -e 's/\chapter{/\chaXter{/g' ${JPYFILENAME}
sed -i '' -e 's/\section{/\chapter{/g' ${JPYFILENAME}
sed -i '' -e 's/\chaXter{/\subsection{/g' ${JPYFILENAME}

# ugly, -i doesnt work here:
sed '28i\
\\usepackage{mathrsfs}  ' ${JPYFILENAME} > tmp-latex
echo renaming: tmp-latex ${JPYFILENAME}; ls -l  tmp-latex ${JPYFILENAME} 
rm ${JPYFILENAME}
mv tmp-latex ${JPYFILENAME}

echo running LATEX
pdflatex book

echo running LATEX , 2nd pass
pdflatex book

cd ../..

