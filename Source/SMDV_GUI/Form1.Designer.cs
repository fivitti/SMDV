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
            this.checkBoxBrakDanych = new System.Windows.Forms.CheckBox();
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
            this.SuspendLayout();
            // 
            // labelSciezkaDoMacierzy
            // 
            this.labelSciezkaDoMacierzy.AutoSize = true;
            this.labelSciezkaDoMacierzy.Location = new System.Drawing.Point(12, 35);
            this.labelSciezkaDoMacierzy.Name = "labelSciezkaDoMacierzy";
            this.labelSciezkaDoMacierzy.Size = new System.Drawing.Size(107, 13);
            this.labelSciezkaDoMacierzy.TabIndex = 0;
            this.labelSciezkaDoMacierzy.Text = "Ścieżka do macierzy:";
            // 
            // labelSciezkaDoLogow
            // 
            this.labelSciezkaDoLogow.AutoSize = true;
            this.labelSciezkaDoLogow.Location = new System.Drawing.Point(12, 61);
            this.labelSciezkaDoLogow.Name = "labelSciezkaDoLogow";
            this.labelSciezkaDoLogow.Size = new System.Drawing.Size(94, 13);
            this.labelSciezkaDoLogow.TabIndex = 1;
            this.labelSciezkaDoLogow.Text = "Ścieżka do logów:";
            // 
            // textBoxSciezkaDoMacierzy
            // 
            this.textBoxSciezkaDoMacierzy.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.textBoxSciezkaDoMacierzy.Location = new System.Drawing.Point(138, 32);
            this.textBoxSciezkaDoMacierzy.Name = "textBoxSciezkaDoMacierzy";
            this.textBoxSciezkaDoMacierzy.Size = new System.Drawing.Size(134, 20);
            this.textBoxSciezkaDoMacierzy.TabIndex = 2;
            this.textBoxSciezkaDoMacierzy.Text = "Macierze\\\\";
            // 
            // textBoxSciezkaDoLogow
            // 
            this.textBoxSciezkaDoLogow.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.textBoxSciezkaDoLogow.Location = new System.Drawing.Point(138, 58);
            this.textBoxSciezkaDoLogow.Name = "textBoxSciezkaDoLogow";
            this.textBoxSciezkaDoLogow.Size = new System.Drawing.Size(134, 20);
            this.textBoxSciezkaDoLogow.TabIndex = 3;
            this.textBoxSciezkaDoLogow.Text = "Logi\\\\";
            // 
            // textBoxSeparator
            // 
            this.textBoxSeparator.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.textBoxSeparator.Location = new System.Drawing.Point(213, 85);
            this.textBoxSeparator.Name = "textBoxSeparator";
            this.textBoxSeparator.Size = new System.Drawing.Size(59, 20);
            this.textBoxSeparator.TabIndex = 4;
            this.textBoxSeparator.Text = ",";
            // 
            // labelSeparator
            // 
            this.labelSeparator.AutoSize = true;
            this.labelSeparator.Location = new System.Drawing.Point(12, 88);
            this.labelSeparator.Name = "labelSeparator";
            this.labelSeparator.Size = new System.Drawing.Size(56, 13);
            this.labelSeparator.TabIndex = 5;
            this.labelSeparator.Text = "Separator:";
            // 
            // textBoxSymbolBrakDanych
            // 
            this.textBoxSymbolBrakDanych.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.textBoxSymbolBrakDanych.Location = new System.Drawing.Point(213, 112);
            this.textBoxSymbolBrakDanych.Name = "textBoxSymbolBrakDanych";
            this.textBoxSymbolBrakDanych.Size = new System.Drawing.Size(59, 20);
            this.textBoxSymbolBrakDanych.TabIndex = 6;
            this.textBoxSymbolBrakDanych.Text = "b/d";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(12, 115);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(112, 13);
            this.label1.TabIndex = 7;
            this.label1.Text = "Symbol braku danych:";
            // 
            // checkBoxCUDA
            // 
            this.checkBoxCUDA.AutoSize = true;
            this.checkBoxCUDA.Location = new System.Drawing.Point(15, 165);
            this.checkBoxCUDA.Name = "checkBoxCUDA";
            this.checkBoxCUDA.Size = new System.Drawing.Size(123, 17);
            this.checkBoxCUDA.TabIndex = 8;
            this.checkBoxCUDA.Text = "Obliczenia na CUDA";
            this.checkBoxCUDA.UseVisualStyleBackColor = true;
            // 
            // checkBoxCPU
            // 
            this.checkBoxCPU.AutoSize = true;
            this.checkBoxCPU.Location = new System.Drawing.Point(15, 188);
            this.checkBoxCPU.Name = "checkBoxCPU";
            this.checkBoxCPU.Size = new System.Drawing.Size(115, 17);
            this.checkBoxCPU.TabIndex = 9;
            this.checkBoxCPU.Text = "Obliczenia na CPU";
            this.checkBoxCPU.UseVisualStyleBackColor = true;
            // 
            // checkBoxBrakDanych
            // 
            this.checkBoxBrakDanych.AutoSize = true;
            this.checkBoxBrakDanych.Checked = true;
            this.checkBoxBrakDanych.CheckState = System.Windows.Forms.CheckState.Checked;
            this.checkBoxBrakDanych.Enabled = false;
            this.checkBoxBrakDanych.Location = new System.Drawing.Point(15, 257);
            this.checkBoxBrakDanych.Name = "checkBoxBrakDanych";
            this.checkBoxBrakDanych.Size = new System.Drawing.Size(126, 17);
            this.checkBoxBrakDanych.TabIndex = 10;
            this.checkBoxBrakDanych.Text = "Niesprawdzone dane";
            this.checkBoxBrakDanych.UseVisualStyleBackColor = true;
            // 
            // checkedListBoxMacierze
            // 
            this.checkedListBoxMacierze.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.checkedListBoxMacierze.CheckOnClick = true;
            this.checkedListBoxMacierze.FormattingEnabled = true;
            this.checkedListBoxMacierze.Location = new System.Drawing.Point(15, 280);
            this.checkedListBoxMacierze.MultiColumn = true;
            this.checkedListBoxMacierze.Name = "checkedListBoxMacierze";
            this.checkedListBoxMacierze.Size = new System.Drawing.Size(257, 124);
            this.checkedListBoxMacierze.Sorted = true;
            this.checkedListBoxMacierze.TabIndex = 11;
            // 
            // checkBoxKonsola
            // 
            this.checkBoxKonsola.AutoSize = true;
            this.checkBoxKonsola.Checked = true;
            this.checkBoxKonsola.CheckState = System.Windows.Forms.CheckState.Checked;
            this.checkBoxKonsola.Location = new System.Drawing.Point(15, 211);
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
            this.checkBoxPlik.Location = new System.Drawing.Point(15, 234);
            this.checkBoxPlik.Name = "checkBoxPlik";
            this.checkBoxPlik.Size = new System.Drawing.Size(102, 17);
            this.checkBoxPlik.TabIndex = 16;
            this.checkBoxPlik.Text = "Utworzenie logu";
            this.checkBoxPlik.UseVisualStyleBackColor = true;
            // 
            // textBoxPowtorzenia
            // 
            this.textBoxPowtorzenia.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.textBoxPowtorzenia.Location = new System.Drawing.Point(213, 139);
            this.textBoxPowtorzenia.Name = "textBoxPowtorzenia";
            this.textBoxPowtorzenia.Size = new System.Drawing.Size(59, 20);
            this.textBoxPowtorzenia.TabIndex = 17;
            this.textBoxPowtorzenia.Text = "100";
            // 
            // labelPowtorzenia
            // 
            this.labelPowtorzenia.AutoSize = true;
            this.labelPowtorzenia.Location = new System.Drawing.Point(12, 142);
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
            this.textBoxSciezkaDoProgramu.Size = new System.Drawing.Size(134, 20);
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
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(284, 473);
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
            this.Controls.Add(this.checkBoxBrakDanych);
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
        private System.Windows.Forms.CheckBox checkBoxBrakDanych;
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

    }
}

