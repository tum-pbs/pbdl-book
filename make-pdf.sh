# source this file with "." in a shell

echo
echo Note: first comment out PG chapter in _toc
echo Note: manually quit first latex pass with shift-x 
echo

#cd /Users/thuerey/Dropbox/mbaDevelSelected/pbdl-book/
/Users/thuerey/Library/Python/3.7/bin/jupyter-book build . --builder pdflatex

# fix up latex 

#cd /Users/thuerey/Dropbox/mbaDevelSelected/pbdl-book/
cd _build/latex
export JPYFILENAME=book.tex
rm ${JPYFILENAME}-in.bak
mv ${JPYFILENAME} ${JPYFILENAME}-in.bak
iconv -c -f utf-8 -t ascii ${JPYFILENAME}-in.bak > ${JPYFILENAME}

echo running SED
# ugly fix for double {{name}.jpg} includes, eg
# \sphinxincludegraphics{{physics-based-deep-learning-overview}.jpg}
# \sphinxincludegraphics[height=240\sphinxpxdimen]{{overview-pano}.jpg}
sed -i '' -e 's/sphinxincludegraphics{{/sphinxincludegraphics{/g' ${JPYFILENAME}
sed -i '' -e 's/}.png}/.png}/g' ${JPYFILENAME}
sed -i '' -e 's/}.jpg}/.jpg}/g' ${JPYFILENAME}
sed -i '' -e 's/}.jpeg}/.jpeg}/g' ${JPYFILENAME}
sed -i '' -e 's/sphinxpxdimen]{{/sphinxpxdimen]{/g' ${JPYFILENAME}

# dirty fix for chapters
# note, keep chapters? (chaXter) only move all other sections one level "up"?
# sed -i '' -e 's///g' ${JPYFILENAME}
sed -i '' -e 's/\\chapter{/\\chaXter{/g' ${JPYFILENAME}
sed -i '' -e 's/\\section{/\\chapter{/g' ${JPYFILENAME}
sed -i '' -e 's/\\chaXter{/\\subsection{/g' ${JPYFILENAME}

# include mathrsfs package for mathscr font
# ugly, -i doesnt work here:
sed '28i\
\\usepackage{mathrsfs}  ' ${JPYFILENAME} > tmp-latex
echo renaming: tmp-latex ${JPYFILENAME}; ls -l  tmp-latex ${JPYFILENAME} 
rm ${JPYFILENAME}
mv tmp-latex ${JPYFILENAME}

# finally done

echo running LATEX
pdflatex book

echo running LATEX , 2nd pass
pdflatex book

cd ../..

