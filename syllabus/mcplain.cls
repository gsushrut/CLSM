% mcplain.cls -- simple format based on early MC LaTeX template document
%
% Main tasks:
% -- loads geometry package for page formatting (and sets 1in margins default)
% -- provides paragraph formatting commands
% -- provides page numbering with ticks
%
% Commands:
%   \setlinespacing{n}
%   \setparspacing{n}
%   \setparindentation{n}
%
% Page styles:
%   plaintick (default)
%
% Mark Caprio
% Orig. 3/13/06
% Using code from old MC documents and mcslide

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Class prologue
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}[1996/06/01]
\ProvidesClass{mcplain}[2009/08/28 v0.0 (Mark Caprio)]

\LoadClassWithOptions{article}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Packages for user convenience
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{color}
\RequirePackage{graphicx}
\RequirePackage{wpindent}

\RequirePackage[paper=letterpaper]{geometry}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Page numbering with tick
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{fancyhdr}
\fancypagestyle{plaintick}{
\fancyhf{}
\fancyhf[FC]{-~\thepage~-}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Formatting declarations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Float placement
\renewcommand{\topfraction}{.99}
\renewcommand{\bottomfraction }{0.99}
\renewcommand{\textfraction }{0.01}
%%\setcounter{\topnumber}{1}

% Widow/orphan control
\widowpenalty=10000
\clubpenalty=10000

% Line breaking
\sloppy

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Formatting commands
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newlength{\MCP@baselineskip}
\setlength{\MCP@baselineskip}{\baselineskip} 
\newlength{\MCP@parindent}
\setlength{\MCP@parindent}{\parindent} 

\newcommand*{\setlinespacing}[1]{\setlength{\baselineskip}{#1\MCP@baselineskip}}
% Note: This almost could be accomplished with \renewcommand{\baselinestretch}{n},
% instead, except when the environment messes up \baselineskip.
\newcommand*{\setparspacing}[1]{\setlength{\parskip}{#1\MCP@baselineskip}}
\newcommand*{\setparindentation}[1]{\setlength{\parindent}{#1\MCP@parindent}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Default formatting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% \setlinespacing{1}
%% \setparspacing{0}
%% \setparindentation{1}
\geometry{hmargin=1in,vmargin=1in}
\pagestyle{plaintick}
\sloppy
