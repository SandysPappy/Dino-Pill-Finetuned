set nocompatible
filetype off

let g:jedi#force_py_version = 3

call plug#begin()
Plug 'junegunn/seoul256.vim'
Plug 'junegunn/fzf'
Plug 'junegunn/fzf.vim'
Plug 'itchyny/lightline.vim'
Plug 'preservim/nerdtree'
call plug#end()

" git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
Plugin 'davidhalter/jedi-vim'
call vundle#end()

map <C-o> :NERDTreeToggle<CR>
"Start NERDTree and put the cursor back in the other window."
"autocmd VimEnter * NERDTree | wincmd p
"closes NERDTree if its the only open open"
autocmd BufEnter * if tabpagenr('$') == 1 && winnr('$') == 1 && exists('b:NERDTree') && b:NERDTree.isTabTree() | quit | endif
let g:seoul256_background = 236
colo seoul256
set background=dark
set t_vb=
filetype plugin indent on
set number
set wrap
set smartcase
set hlsearch
set linebreak
set scrolloff=1
set laststatus=2
syntax enable
autocmd Filetype python setlocal softtabstop=4 tabstop=4 shiftwidth=4 expandtab
set autoread

set visualbell t_vb=
if has("autocmd") && has("gui")
    au GUIEnter * set t_vb=
endif
