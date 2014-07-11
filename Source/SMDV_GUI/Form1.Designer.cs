namespace GUIforSMDV
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.labelSciezkaDoMacierzy = new System.Windows.Forms.Label();
            this.labelSciezkaDoLogow = new System.Windows.Forms.Label();
            this.textBoxSciezkaDoMacierzy = new System.Windows.Forms.TextBox();
            this.textBoxSciezkaDoLogow = new System.Windows.Forms.TextBox();
            this.textBoxSeparator = new System.Windows.Forms.TextBox();
            this.labelSeparator = new System.Windows.Forms.Label();
            this.textBoxSymbolBrakDanych = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.checkBoxCUDA = new System.Windows.Forms.CheckBox();
            this.checkBoxCPU = new System.Windows.Forms.CheckBox();
            this.checkedListBoxMacierze = new System.Windows.Forms.CheckedListBox();
            this.checkBoxKonsola = new System.Windows.Forms.CheckBox();
            this.buttonStart = new System.Windows.Forms.Button();
            this.labelLog = new System.Windows.Forms.Label();
            this.checkBoxPlik = new System.Windows.Forms.CheckBox();
            this.textBoxPowtorzenia = new System.Windows.Forms.TextBox();
            this.labelPowtorzenia = new System.Windows.Forms.Label();
            this.textBoxSciezkaDoProgramu = new System.Windows.Forms.TextBox();
            this.labelSciezkaProgramu = new System.Windows.Forms.Label();
            this.buttonSzukaj = new System.Windows.Forms.Button();
            this.buttonWyborSciezkiSMDV = new System.Windows.Forms.Button();
            this.folderBrowserDialog1 = new System.Windows.Forms.FolderBrowserDialog();
            this.buttonSciezkaDoMacierzy = new System.Windows.Forms.Button();
            this.buttonSciezkaDoLogow = new System.Windows.Forms.Button();
            this.textBoxSciezkaDoPythonSMDV = new System.Windows.Forms.TextBox();
            this.labelSciezkaDoPythonSMDV = new System.Windows.Forms.Label();
            this.buttonSciezkaPySMDV = new System.Windows.Forms.Button();
            this.checkBoxSMDV = new System.Windows.Forms.CheckBox();
            this.checkBoxPySMDV = new System.Windows.Forms.CheckBox();
            this.openFileDialog1 = new System.Windows.Forms.OpenFileDialog();
            this.textBoxPython = new System.Windows.Forms.TextBox();
            this.labelPython = new System.Windows.Forms.Label();
            this.buttonPython = new System.Windows.Forms.Button();
            this.labelRozmiarBloku = new System.Windows.Forms.Label();
            this.textBoxRozmiarBloku = new System.Windows.Forms.TextBox();
            this.textBoxSliceSize = new System.Windows.Forms.TextBox();
            this.labelSliceSize = new System.Windows.Forms.Label();
            this.textBoxWatkiNaWiersz = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.SuspendLayout();
            // 
            // labelSciezkaDoMacierzy
            // 
            this.labelSciezkaDoMacierzy.AutoSize = true;
            this.labelSciezkaDoMacierzy.Location = new System.Drawing.Point(12, 85);
            this.labelSciezkaDoMacierzy.Name = "labelSciezkaDoMacierzy";
            this.labelSciezkaDoMacierzy.Size = new System.Drawing.Size(107, 13);
            this.labelSciezkaDoMacierzy.TabIndex = 0;
            this.labelSciezkaDoMacierzy.Text = "Ścieżka do macierzy:";
            // 
            // labelSciezkaDoLogow
            // 
            this.labelSciezkaDoLogow.AutoSize = true;
            this.labelSciezkaDoLogow.Location = new System.Drawing.Point(12, 111);
            this.labelSciezkaDoLogow.Name = "labelSciezkaDoLogow";
            this.labelSciezkaDoLogow.Size = new System.Drawing.Size(94, 13);
            this.labelSciezkaDoLogow.TabIndex = 1;
            this.labelSciezkaDoLogow.Text = "Ścieżka do logów:";
            // 
            // textBoxSciezkaDoMacierzy
            // 
            this.textBoxSciezkaDoMacierzy.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.textBoxSciezkaDoMacierzy.Location = new System.Drawing.Point(138, 82);
            this.textBoxSciezkaDoMacierzy.Name = "textBoxSciezkaDoMacierzy";
            this.textBoxSciezkaDoMacierzy.Size = new System.Drawing.Size(181, 20);
            this.textBoxSciezkaDoMacierzy.TabIndex = 2;
            this.textBoxSciezkaDoMacierzy.Text = "Macierze\\\\";
            // 
            // textBoxSciezkaDoLogow
            // 
            this.textBoxSciezkaDoLogow.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.textBoxSciezkaDoLogow.Location = new System.Drawing.Point(138, 108);
            this.textBoxSciezkaDoLogow.Name = "textBoxSciezkaDoLogow";
            this.textBoxSciezkaDoLogow.Size = new System.Drawing.Size(181, 20);
            this.textBoxSciezkaDoLogow.TabIndex = 3;
            this.textBoxSciezkaDoLogow.Text = "Logi\\\\";
            // 
            // textBoxSeparator
            // 
            this.textBoxSeparator.Location = new System.Drawing.Point(138, 134);
            this.textBoxSeparator.Name = "textBoxSeparator";
            this.textBoxSeparator.Size = new System.Drawing.Size(59, 20);
            this.textBoxSeparator.TabIndex = 4;
            this.textBoxSeparator.Text = ";";
            // 
            // labelSeparator
            // 
            this.labelSeparator.AutoSize = true;
            this.labelSeparator.Location = new System.Drawing.Point(12, 137);
            this.labelSeparator.Name = "labelSeparator";
            this.labelSeparator.Size = new System.Drawing.Size(56, 13);
            this.labelSeparator.TabIndex = 5;
            this.labelSeparator.Text = "Separator:";
            // 
            // textBoxSymbolBrakDanych
            // 
            this.textBoxSymbolBrakDanych.Location = new System.Drawing.Point(138, 160);
            this.textBoxSymbolBrakDanych.Name = "textBoxSymbolBrakDanych";
            this.textBoxSymbolBrakDanych.Size = new System.Drawing.Size(59, 20);
            this.textBoxSymbolBrakDanych.TabIndex = 6;
            this.textBoxSymbolBrakDanych.Text = "b/d";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(12, 163);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(112, 13);
            this.label1.TabIndex = 7;
            this.label1.Text = "Symbol braku danych:";
            // 
            // checkBoxCUDA
            // 
            this.checkBoxCUDA.AutoSize = true;
            this.checkBoxCUDA.Location = new System.Drawing.Point(15, 286);
            this.checkBoxCUDA.Name = "checkBoxCUDA";
            this.checkBoxCUDA.Size = new System.Drawing.Size(123, 17);
            this.checkBoxCUDA.TabIndex = 8;
            this.checkBoxCUDA.Text = "Obliczenia na CUDA";
            this.checkBoxCUDA.UseVisualStyleBackColor = true;
            // 
            // checkBoxCPU
            // 
            this.checkBoxCPU.AutoSize = true;
            this.checkBoxCPU.Location = new System.Drawing.Point(15, 263);
            this.checkBoxCPU.Name = "checkBoxCPU";
            this.checkBoxCPU.Size = new System.Drawing.Size(115, 17);
            this.checkBoxCPU.TabIndex = 9;
            this.checkBoxCPU.Text = "Obliczenia na CPU";
            this.checkBoxCPU.UseVisualStyleBackColor = true;
            // 
            // checkedListBoxMacierze
            // 
            this.checkedListBoxMacierze.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.checkedListBoxMacierze.CheckOnClick = true;
            this.checkedListBoxMacierze.FormattingEnabled = true;
            this.checkedListBoxMacierze.Location = new System.Drawing.Point(15, 310);
            this.checkedListBoxMacierze.MultiColumn = true;
            this.checkedListBoxMacierze.Name = "checkedListBoxMacierze";
            this.checkedListBoxMacierze.Size = new System.Drawing.Size(334, 94);
            this.checkedListBoxMacierze.Sorted = true;
            this.checkedListBoxMacierze.TabIndex = 11;
            // 
            // checkBoxKonsola
            // 
            this.checkBoxKonsola.AutoSize = true;
            this.checkBoxKonsola.Checked = true;
            this.checkBoxKonsola.CheckState = System.Windows.Forms.CheckState.Checked;
            this.checkBoxKonsola.Location = new System.Drawing.Point(15, 217);
            this.checkBoxKonsola.Name = "checkBoxKonsola";
            this.checkBoxKonsola.Size = new System.Drawing.Size(132, 17);
            this.checkBoxKonsola.TabIndex = 12;
            this.checkBoxKonsola.Text = "Komunikaty na konsoli";
            this.checkBoxKonsola.UseVisualStyleBackColor = true;
            // 
            // buttonStart
            // 
            this.buttonStart.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.buttonStart.Location = new System.Drawing.Point(15, 410);
            this.buttonStart.Name = "buttonStart";
            this.buttonStart.Size = new System.Drawing.Size(75, 23);
            this.buttonStart.TabIndex = 13;
            this.buttonStart.Text = "Start";
            this.buttonStart.UseVisualStyleBackColor = true;
            this.buttonStart.Click += new System.EventHandler(this.buttonStart_Click);
            // 
            // labelLog
            // 
            this.labelLog.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.labelLog.AutoSize = true;
            this.labelLog.Location = new System.Drawing.Point(15, 440);
            this.labelLog.Name = "labelLog";
            this.labelLog.Size = new System.Drawing.Size(90, 13);
            this.labelLog.TabIndex = 14;
            this.labelLog.Text = "Brak błędów GUI";
            // 
            // checkBoxPlik
            // 
            this.checkBoxPlik.AutoSize = true;
            this.checkBoxPlik.Checked = true;
            this.checkBoxPlik.CheckState = System.Windows.Forms.CheckState.Checked;
            this.checkBoxPlik.Location = new System.Drawing.Point(15, 240);
            this.checkBoxPlik.Name = "checkBoxPlik";
            this.checkBoxPlik.Size = new System.Drawing.Size(102, 17);
            this.checkBoxPlik.TabIndex = 16;
            this.checkBoxPlik.Text = "Utworzenie logu";
            this.checkBoxPlik.UseVisualStyleBackColor = true;
            // 
            // textBoxPowtorzenia
            // 
            this.textBoxPowtorzenia.Location = new System.Drawing.Point(138, 186);
            this.textBoxPowtorzenia.Name = "textBoxPowtorzenia";
            this.textBoxPowtorzenia.Size = new System.Drawing.Size(59, 20);
            this.textBoxPowtorzenia.TabIndex = 17;
            this.textBoxPowtorzenia.Text = "100";
            // 
            // labelPowtorzenia
            // 
            this.labelPowtorzenia.AutoSize = true;
            this.labelPowtorzenia.Location = new System.Drawing.Point(12, 189);
            this.labelPowtorzenia.Name = "labelPowtorzenia";
            this.labelPowtorzenia.Size = new System.Drawing.Size(68, 13);
            this.labelPowtorzenia.TabIndex = 18;
            this.labelPowtorzenia.Text = "Powtórzenia:";
            // 
            // textBoxSciezkaDoProgramu
            // 
            this.textBoxSciezkaDoProgramu.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.textBoxSciezkaDoProgramu.Location = new System.Drawing.Point(138, 6);
            this.textBoxSciezkaDoProgramu.Name = "textBoxSciezkaDoProgramu";
            this.textBoxSciezkaDoProgramu.Size = new System.Drawing.Size(181, 20);
            this.textBoxSciezkaDoProgramu.TabIndex = 19;
            // 
            // labelSciezkaProgramu
            // 
            this.labelSciezkaProgramu.AutoSize = true;
            this.labelSciezkaProgramu.Location = new System.Drawing.Point(12, 9);
            this.labelSciezkaProgramu.Name = "labelSciezkaProgramu";
            this.labelSciezkaProgramu.Size = new System.Drawing.Size(97, 13);
            this.labelSciezkaProgramu.TabIndex = 20;
            this.labelSciezkaProgramu.Text = "Ścieżka do SMDV:";
            // 
            // buttonSzukaj
            // 
            this.buttonSzukaj.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left)));
            this.buttonSzukaj.Location = new System.Drawing.Point(96, 410);
            this.buttonSzukaj.Name = "buttonSzukaj";
            this.buttonSzukaj.Size = new System.Drawing.Size(75, 23);
            this.buttonSzukaj.TabIndex = 15;
            this.buttonSzukaj.Text = "Szukaj";
            this.buttonSzukaj.UseVisualStyleBackColor = true;
            this.buttonSzukaj.Click += new System.EventHandler(this.buttonSzukaj_Click);
            // 
            // buttonWyborSciezkiSMDV
            // 
            this.buttonWyborSciezkiSMDV.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.buttonWyborSciezkiSMDV.Location = new System.Drawing.Point(325, 4);
            this.buttonWyborSciezkiSMDV.Name = "buttonWyborSciezkiSMDV";
            this.buttonWyborSciezkiSMDV.Size = new System.Drawing.Size(24, 23);
            this.buttonWyborSciezkiSMDV.TabIndex = 21;
            this.buttonWyborSciezkiSMDV.Text = "...";
            this.buttonWyborSciezkiSMDV.UseVisualStyleBackColor = true;
            this.buttonWyborSciezkiSMDV.Click += new System.EventHandler(this.buttonWyborSciezkiSMDV_Click);
            // 
            // buttonSciezkaDoMacierzy
            // 
            this.buttonSciezkaDoMacierzy.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.buttonSciezkaDoMacierzy.Location = new System.Drawing.Point(325, 80);
            this.buttonSciezkaDoMacierzy.Name = "buttonSciezkaDoMacierzy";
            this.buttonSciezkaDoMacierzy.Size = new System.Drawing.Size(24, 23);
            this.buttonSciezkaDoMacierzy.TabIndex = 22;
            this.buttonSciezkaDoMacierzy.Text = "...";
            this.buttonSciezkaDoMacierzy.UseVisualStyleBackColor = true;
            this.buttonSciezkaDoMacierzy.Click += new System.EventHandler(this.buttonSciezkaDoMacierzy_Click);
            // 
            // buttonSciezkaDoLogow
            // 
            this.buttonSciezkaDoLogow.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.buttonSciezkaDoLogow.Location = new System.Drawing.Point(325, 106);
            this.buttonSciezkaDoLogow.Name = "buttonSciezkaDoLogow";
            this.buttonSciezkaDoLogow.Size = new System.Drawing.Size(24, 23);
            this.buttonSciezkaDoLogow.TabIndex = 23;
            this.buttonSciezkaDoLogow.Text = "...";
            this.buttonSciezkaDoLogow.UseVisualStyleBackColor = true;
            this.buttonSciezkaDoLogow.Click += new System.EventHandler(this.buttonSciezkaDoLogow_Click);
            // 
            // textBoxSciezkaDoPythonSMDV
            // 
            this.textBoxSciezkaDoPythonSMDV.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.textBoxSciezkaDoPythonSMDV.Location = new System.Drawing.Point(138, 30);
            this.textBoxSciezkaDoPythonSMDV.Name = "textBoxSciezkaDoPythonSMDV";
            this.textBoxSciezkaDoPythonSMDV.Size = new System.Drawing.Size(181, 20);
            this.textBoxSciezkaDoPythonSMDV.TabIndex = 24;
            // 
            // labelSciezkaDoPythonSMDV
            // 
            this.labelSciezkaDoPythonSMDV.AutoSize = true;
            this.labelSciezkaDoPythonSMDV.Location = new System.Drawing.Point(12, 33);
            this.labelSciezkaDoPythonSMDV.Name = "labelSciezkaDoPythonSMDV";
            this.labelSciezkaDoPythonSMDV.Size = new System.Drawing.Size(109, 13);
            this.labelSciezkaDoPythonSMDV.TabIndex = 25;
            this.labelSciezkaDoPythonSMDV.Text = "Ścieżka do PySMDV:";
            // 
            // buttonSciezkaPySMDV
            // 
            this.buttonSciezkaPySMDV.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.buttonSciezkaPySMDV.Location = new System.Drawing.Point(325, 28);
            this.buttonSciezkaPySMDV.Name = "buttonSciezkaPySMDV";
            this.buttonSciezkaPySMDV.Size = new System.Drawing.Size(24, 23);
            this.buttonSciezkaPySMDV.TabIndex = 26;
            this.buttonSciezkaPySMDV.Text = "...";
            this.buttonSciezkaPySMDV.UseVisualStyleBackColor = true;
            this.buttonSciezkaPySMDV.Click += new System.EventHandler(this.buttonSciezkaPySMDV_Click);
            // 
            // checkBoxSMDV
            // 
            this.checkBoxSMDV.AutoSize = true;
            this.checkBoxSMDV.Location = new System.Drawing.Point(164, 217);
            this.checkBoxSMDV.Name = "checkBoxSMDV";
            this.checkBoxSMDV.Size = new System.Drawing.Size(78, 17);
            this.checkBoxSMDV.TabIndex = 27;
            this.checkBoxSMDV.Text = "Użyj CUSP";
            this.checkBoxSMDV.UseVisualStyleBackColor = true;
            // 
            // checkBoxPySMDV
            // 
            this.checkBoxPySMDV.AutoSize = true;
            this.checkBoxPySMDV.Location = new System.Drawing.Point(164, 241);
            this.checkBoxPySMDV.Name = "checkBoxPySMDV";
            this.checkBoxPySMDV.Size = new System.Drawing.Size(92, 17);
            this.checkBoxPySMDV.TabIndex = 28;
            this.checkBoxPySMDV.Text = "Użyj PySMDV";
            this.checkBoxPySMDV.UseVisualStyleBackColor = true;
            // 
            // openFileDialog1
            // 
            this.openFileDialog1.FileName = "openFileDialog1";
            // 
            // textBoxPython
            // 
            this.textBoxPython.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.textBoxPython.Location = new System.Drawing.Point(138, 56);
            this.textBoxPython.Name = "textBoxPython";
            this.textBoxPython.Size = new System.Drawing.Size(181, 20);
            this.textBoxPython.TabIndex = 29;
            // 
            // labelPython
            // 
            this.labelPython.AutoSize = true;
            this.labelPython.Location = new System.Drawing.Point(12, 59);
            this.labelPython.Name = "labelPython";
            this.labelPython.Size = new System.Drawing.Size(119, 13);
            this.labelPython.TabIndex = 30;
            this.labelPython.Text = "Ścieżka do Python.exe:";
            // 
            // buttonPython
            // 
            this.buttonPython.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.buttonPython.Location = new System.Drawing.Point(325, 54);
            this.buttonPython.Name = "buttonPython";
            this.buttonPython.Size = new System.Drawing.Size(24, 23);
            this.buttonPython.TabIndex = 31;
            this.buttonPython.Text = "...";
            this.buttonPython.UseVisualStyleBackColor = true;
            this.buttonPython.Click += new System.EventHandler(this.buttonPython_Click);
            // 
            // labelRozmiarBloku
            // 
            this.labelRozmiarBloku.AutoSize = true;
            this.labelRozmiarBloku.Location = new System.Drawing.Point(203, 137);
            this.labelRozmiarBloku.Name = "labelRozmiarBloku";
            this.labelRozmiarBloku.Size = new System.Drawing.Size(77, 13);
            this.labelRozmiarBloku.TabIndex = 32;
            this.labelRozmiarBloku.Text = "Rozmiar bloku:";
            // 
            // textBoxRozmiarBloku
            // 
            this.textBoxRozmiarBloku.Location = new System.Drawing.Point(286, 134);
            this.textBoxRozmiarBloku.Name = "textBoxRozmiarBloku";
            this.textBoxRozmiarBloku.Size = new System.Drawing.Size(33, 20);
            this.textBoxRozmiarBloku.TabIndex = 33;
            this.textBoxRozmiarBloku.Text = "128";
            // 
            // textBoxSliceSize
            // 
            this.textBoxSliceSize.Location = new System.Drawing.Point(286, 160);
            this.textBoxSliceSize.Name = "textBoxSliceSize";
            this.textBoxSliceSize.Size = new System.Drawing.Size(33, 20);
            this.textBoxSliceSize.TabIndex = 33;
            this.textBoxSliceSize.Text = "2";
            // 
            // labelSliceSize
            // 
            this.labelSliceSize.AutoSize = true;
            this.labelSliceSize.Location = new System.Drawing.Point(203, 163);
            this.labelSliceSize.Name = "labelSliceSize";
            this.labelSliceSize.Size = new System.Drawing.Size(54, 13);
            this.labelSliceSize.TabIndex = 34;
            this.labelSliceSize.Text = "Slice size:";
            // 
            // textBoxWatkiNaWiersz
            // 
            this.textBoxWatkiNaWiersz.Location = new System.Drawing.Point(286, 186);
            this.textBoxWatkiNaWiersz.Name = "textBoxWatkiNaWiersz";
            this.textBoxWatkiNaWiersz.Size = new System.Drawing.Size(33, 20);
            this.textBoxWatkiNaWiersz.TabIndex = 35;
            this.textBoxWatkiNaWiersz.Text = "2";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(203, 189);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(85, 13);
            this.label2.TabIndex = 36;
            this.label2.Text = "Wątki na wiersz:";
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(361, 473);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.textBoxWatkiNaWiersz);
            this.Controls.Add(this.labelSliceSize);
            this.Controls.Add(this.textBoxSliceSize);
            this.Controls.Add(this.textBoxRozmiarBloku);
            this.Controls.Add(this.labelRozmiarBloku);
            this.Controls.Add(this.buttonPython);
            this.Controls.Add(this.labelPython);
            this.Controls.Add(this.textBoxPython);
            this.Controls.Add(this.checkBoxPySMDV);
            this.Controls.Add(this.checkBoxSMDV);
            this.Controls.Add(this.buttonSciezkaPySMDV);
            this.Controls.Add(this.labelSciezkaDoPythonSMDV);
            this.Controls.Add(this.textBoxSciezkaDoPythonSMDV);
            this.Controls.Add(this.buttonSciezkaDoLogow);
            this.Controls.Add(this.buttonSciezkaDoMacierzy);
            this.Controls.Add(this.buttonWyborSciezkiSMDV);
            this.Controls.Add(this.labelSciezkaProgramu);
            this.Controls.Add(this.textBoxSciezkaDoProgramu);
            this.Controls.Add(this.labelPowtorzenia);
            this.Controls.Add(this.textBoxPowtorzenia);
            this.Controls.Add(this.checkBoxPlik);
            this.Controls.Add(this.buttonSzukaj);
            this.Controls.Add(this.labelLog);
            this.Controls.Add(this.buttonStart);
            this.Controls.Add(this.checkBoxKonsola);
            this.Controls.Add(this.checkedListBoxMacierze);
            this.Controls.Add(this.checkBoxCPU);
            this.Controls.Add(this.checkBoxCUDA);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.textBoxSymbolBrakDanych);
            this.Controls.Add(this.labelSeparator);
            this.Controls.Add(this.textBoxSeparator);
            this.Controls.Add(this.textBoxSciezkaDoLogow);
            this.Controls.Add(this.textBoxSciezkaDoMacierzy);
            this.Controls.Add(this.labelSciezkaDoLogow);
            this.Controls.Add(this.labelSciezkaDoMacierzy);
            this.Name = "Form1";
            this.Text = "SMDV";
            this.Load += new System.EventHandler(this.Form1_Load);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label labelSciezkaDoMacierzy;
        private System.Windows.Forms.Label labelSciezkaDoLogow;
        private System.Windows.Forms.TextBox textBoxSciezkaDoMacierzy;
        private System.Windows.Forms.TextBox textBoxSciezkaDoLogow;
        private System.Windows.Forms.TextBox textBoxSeparator;
        private System.Windows.Forms.Label labelSeparator;
        private System.Windows.Forms.TextBox textBoxSymbolBrakDanych;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.CheckBox checkBoxCUDA;
        private System.Windows.Forms.CheckBox checkBoxCPU;
        private System.Windows.Forms.CheckedListBox checkedListBoxMacierze;
        private System.Windows.Forms.CheckBox checkBoxKonsola;
        private System.Windows.Forms.Button buttonStart;
        private System.Windows.Forms.Label labelLog;
        private System.Windows.Forms.CheckBox checkBoxPlik;
        private System.Windows.Forms.TextBox textBoxPowtorzenia;
        private System.Windows.Forms.Label labelPowtorzenia;
        private System.Windows.Forms.TextBox textBoxSciezkaDoProgramu;
        private System.Windows.Forms.Label labelSciezkaProgramu;
        private System.Windows.Forms.Button buttonSzukaj;
        private System.Windows.Forms.Button buttonWyborSciezkiSMDV;
        private System.Windows.Forms.FolderBrowserDialog folderBrowserDialog1;
        private System.Windows.Forms.Button buttonSciezkaDoMacierzy;
        private System.Windows.Forms.Button buttonSciezkaDoLogow;
        private System.Windows.Forms.TextBox textBoxSciezkaDoPythonSMDV;
        private System.Windows.Forms.Label labelSciezkaDoPythonSMDV;
        private System.Windows.Forms.Button buttonSciezkaPySMDV;
        private System.Windows.Forms.CheckBox checkBoxSMDV;
        private System.Windows.Forms.CheckBox checkBoxPySMDV;
        private System.Windows.Forms.OpenFileDialog openFileDialog1;
        private System.Windows.Forms.TextBox textBoxPython;
        private System.Windows.Forms.Label labelPython;
        private System.Windows.Forms.Button buttonPython;
        private System.Windows.Forms.Label labelRozmiarBloku;
        private System.Windows.Forms.TextBox textBoxRozmiarBloku;
        private System.Windows.Forms.TextBox textBoxSliceSize;
        private System.Windows.Forms.Label labelSliceSize;
        private System.Windows.Forms.TextBox textBoxWatkiNaWiersz;
        private System.Windows.Forms.Label label2;

    }
}

