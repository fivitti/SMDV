//TO DO: Dodac informacje o ograniczeniu liczby macierzy

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.IO;
using System.Diagnostics;
using GUIforSMDV.Properties;



namespace GUIforSMDV
{
     
    public partial class Form1 : Form
    {
         
        public Form1()
        {
            InitializeComponent();
            this.FormClosing += new FormClosingEventHandler(Form1_FormClosing);
        }

        private void Form1_Load(object sender, EventArgs e)
        {
                #region odczyt ustawień 
                textBoxSciezkaDoProgramu.Text = Settings.Default.SciezkaDoSMDV;
                textBoxSciezkaDoPythonSMDV.Text = Settings.Default.SciezkaDoPySMDV;
                textBoxSciezkaDoMacierzy.Text = Settings.Default.SciezkaDoMacierzy;
                textBoxSciezkaDoLogow.Text = Settings.Default.SciezkaDoLogow;
                textBoxSeparator.Text = Settings.Default.Separator;
                textBoxSymbolBrakDanych.Text = Settings.Default.SymbolBrakDanych;
                textBoxPowtorzenia.Text = Settings.Default.Powtorzenia;
                checkBoxKonsola.Checked = Settings.Default.KomunikatyNaKonsoli;
                checkBoxPlik.Checked = Settings.Default.UtworzenieLogu;
                checkBoxCPU.Checked = Settings.Default.ObliczeniaNaCPU;
                checkBoxCUDA.Checked = Settings.Default.ObliczeniaNaCUDA;
                checkBoxSMDV.Checked = Settings.Default.ObliczeniaSMDV;
                checkBoxPySMDV.Checked = Settings.Default.ObliczeniaPySMDV;
                textBoxPython.Text = Settings.Default.PythonExe;
                textBoxRozmiarBloku.Text = Settings.Default.RozmiarBloku;
                textBoxSliceSize.Text = Settings.Default.SliceSize;
                textBoxWatkiNaWiersz.Text = Settings.Default.WatkiNaWiersz;

                szukajMacierzy(Settings.Default.Macierze);                           //Przeszukuje katalog z macierzami pod kątem plików .mtx

                #endregion    
        }
        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            #region zapis ustawień
            Settings.Default.SciezkaDoSMDV = textBoxSciezkaDoProgramu.Text;
            Settings.Default.SciezkaDoPySMDV = textBoxSciezkaDoPythonSMDV.Text;
            Settings.Default.SciezkaDoMacierzy = textBoxSciezkaDoMacierzy.Text;
            Settings.Default.SciezkaDoLogow = textBoxSciezkaDoLogow.Text;
            Settings.Default.Separator = textBoxSeparator.Text;
            Settings.Default.SymbolBrakDanych = textBoxSymbolBrakDanych.Text;
            Settings.Default.Powtorzenia = textBoxPowtorzenia.Text;
            Settings.Default.KomunikatyNaKonsoli = checkBoxKonsola.Checked;
            Settings.Default.UtworzenieLogu = checkBoxPlik.Checked;
            Settings.Default.ObliczeniaNaCPU = checkBoxCPU.Checked;
            Settings.Default.ObliczeniaNaCUDA = checkBoxCUDA.Checked;
            Settings.Default.ObliczeniaSMDV = checkBoxSMDV.Checked;
            Settings.Default.ObliczeniaPySMDV = checkBoxPySMDV.Checked;
            Settings.Default.PythonExe = textBoxPython.Text;
            Settings.Default.RozmiarBloku = textBoxRozmiarBloku.Text;
            Settings.Default.SliceSize = textBoxSliceSize.Text;
            Settings.Default.WatkiNaWiersz = textBoxWatkiNaWiersz.Text;

            System.Collections.Specialized.StringCollection macierzeZaznaczone = new System.Collections.Specialized.StringCollection();
            foreach (var item in checkedListBoxMacierze.CheckedItems)
            {
                macierzeZaznaczone.Add(item.ToString());
            }
            Settings.Default.Macierze = macierzeZaznaczone;

            Settings.Default.Save();         
            #endregion
        }

        private void buttonStart_Click(object sender, EventArgs e)
        {
            #region obsluga błędów
            labelLog.Text = "Brak błędów.";
            try
            {
                TextBox[] polaTekstowe = new TextBox[] {textBoxSciezkaDoProgramu, textBoxSciezkaDoLogow, textBoxSciezkaDoMacierzy, textBoxSciezkaDoPythonSMDV, textBoxPython, textBoxSeparator, textBoxSymbolBrakDanych};
                TextBox[] polaLiczbowe = new TextBox[] {textBoxPowtorzenia, textBoxRozmiarBloku, textBoxSliceSize, textBoxWatkiNaWiersz};
                bool czyBlad = false;

                foreach (var item in polaTekstowe)
                {
                    item.BackColor = Color.FromKnownColor(KnownColor.Window);
                    if (item.Text.Trim() == "")
                    {
                        item.BackColor = Color.Red;
                        czyBlad = true;
                    }
                }
                foreach (var item in polaLiczbowe)
                {
                    item.BackColor = Color.FromKnownColor(KnownColor.Window);
                    int liczba = 0;
                    bool czy = Int32.TryParse(item.Text, out liczba);
                    if (czy == false || liczba < 1)
                    {
                        item.BackColor = Color.Red;
                        czyBlad = true;
                    }
                }
                if (czyBlad)
                    throw new InvalidDataException("Błąd: Nieprawidłowy format danych.");
                else if (!(checkBoxCPU.Checked || checkBoxCUDA.Checked))
                {
                    throw new InvalidDataException("Błąd: Brak sposobu wykonania obliczeń.");
                }
                else if (!(checkBoxKonsola.Checked || checkBoxPlik.Checked))
                {
                    throw new InvalidDataException("Błąd: Program nic nie zwraca.");
                }
                else if (checkedListBoxMacierze.CheckedItems.Count < 1)
                {
                    throw new InvalidDataException("Błąd: Brak macierzy.");
                }
                else if (checkBoxPySMDV.Checked && !checkBoxCUDA.Checked)
                {
                    throw new InvalidDataException("Błąd: Obliczenia PySMDV dostępne tylko na CUDA.");
                }
            }
            catch (InvalidDataException ide)
            {
                labelLog.Text = ide.Message;
                return;
            }
            
            #endregion
                              
            if (checkBoxSMDV.Checked == true)
            {
                string wyjscie = string.Format("{0} {1} {2} {3} {4} \"{5}\\\\\" \"{6}\\\\\" {7} {8} {9} {10}",
                                            checkBoxCUDA.Checked,           //0
                                            checkBoxCPU.Checked,            //1
                                            checkBoxKonsola.Checked,        //2
                                            checkBoxPlik.Checked,           //3
                                            textBoxRozmiarBloku.Text,       //4
                                            textBoxSciezkaDoMacierzy.Text,  //5
                                            textBoxSciezkaDoLogow.Text,     //6
                                            textBoxSeparator.Text,          //7
                                            textBoxSymbolBrakDanych.Text,   //8
                                            textBoxPowtorzenia.Text,        //9
                                            checkedElementsInCheckedListToString(checkedListBoxMacierze));//10    
                try
                {
                    Process myProcess = new Process();
                    myProcess.StartInfo.FileName = textBoxSciezkaDoProgramu.Text;
                    //Do not receive an event when the process exits.
                    myProcess.EnableRaisingEvents = false;
                    // Parameters
                    myProcess.StartInfo.Arguments = wyjscie;
                    // Modify the following to hide / show the window
                    myProcess.StartInfo.CreateNoWindow = false;
                    myProcess.StartInfo.UseShellExecute = true;
                    myProcess.StartInfo.WindowStyle = ProcessWindowStyle.Normal;
                    myProcess.Start();
                }
                catch (Exception fe)
                {
                    labelLog.Text = "Błąd: Brak pliku programu SMDV.";
                    return;
                }
            }
            if (checkBoxPySMDV.Checked == true)
            {
                string wyjscie = string.Format("{0} {1} {2} \"{3}\\\\\" \"{4}\\\\\" {5} {6} {7} {8} {9} {10}",
                                            checkBoxKonsola.Checked,        //0
                                            checkBoxPlik.Checked,           //1
                                            textBoxRozmiarBloku.Text,       //2
                                            textBoxSciezkaDoMacierzy.Text,  //3
                                            textBoxSciezkaDoLogow.Text,     //4
                                            textBoxSeparator.Text,          //5
                                            textBoxSymbolBrakDanych.Text,   //6
                                            textBoxPowtorzenia.Text,        //7
                                            textBoxSliceSize.Text,          //8
                                            textBoxWatkiNaWiersz.Text,      //9
                                            checkedElementsInCheckedListToString(checkedListBoxMacierze));//10
                                            
                try
                {
                    Process myProcess = new Process();
                    myProcess.StartInfo.FileName = textBoxPython.Text;
                    //Do not receive an event when the process exits.
                    myProcess.EnableRaisingEvents = false;
                    // Parameters
                    myProcess.StartInfo.Arguments = string.Format("\"{0}\" {1}", textBoxSciezkaDoPythonSMDV.Text, wyjscie);
                    // Modify the following to hide / show the window
                    myProcess.StartInfo.CreateNoWindow = false;
                    myProcess.StartInfo.UseShellExecute = true;
                    myProcess.StartInfo.WindowStyle = ProcessWindowStyle.Normal;
                    myProcess.Start();
                }
                catch (Exception fe)
                {
                    labelLog.Text = "Błąd: Brak pliku programu PySMDV. "+fe.Message;
                    return;
                }
            }
        }

        

        private void buttonSzukaj_Click(object sender, EventArgs e)
        {
            szukajMacierzy();
        }

        private void szukajMacierzy(System.Collections.Specialized.StringCollection doZaznaczenia = null)                                               //TO DO: Funkcja do poprawienia
        {
            string sciezka = textBoxSciezkaDoMacierzy.Text;
            List<string> macierzePliki;
            System.Collections.Specialized.StringCollection macierzeZaznaczone;
            if (doZaznaczenia == null)
            {
                macierzeZaznaczone = new System.Collections.Specialized.StringCollection();

                foreach (var item in checkedListBoxMacierze.CheckedItems)
                {
                    macierzeZaznaczone.Add(item.ToString());
                }
            }
            else
                macierzeZaznaczone = doZaznaczenia;

            try
            {
                macierzePliki = Directory.GetFiles(sciezka, "*.mtx").ToList<string>();
            }
            catch (DirectoryNotFoundException e)
            {
                labelLog.Text = "Folder z macierzami nie istnieje.";
                return;
            }

            #region normalizacja
            for (int i = 0; i < macierzePliki.Count; ++i)
            {
                macierzePliki[i] = macierzePliki[i].Replace(sciezka+"\\", "");
            }
            #endregion

            checkedListBoxMacierze.Items.Clear();

            foreach (string macierz in macierzePliki)
            {
                if (stringWLiscie(macierz, macierzeZaznaczone))
                {
                    checkedListBoxMacierze.Items.Add(macierz, true);
                    macierzeZaznaczone.Remove(macierz);
                }
                else
                    checkedListBoxMacierze.Items.Add(macierz, false);
            }
        }

        private bool stringWLiscie(string s, System.Collections.Specialized.StringCollection lista)
        {
            foreach (string item in lista)
            {
                if (s == item)
                    return true;      
            }
            return false;
        }

        private string checkedElementsInCheckedListToString(CheckedListBox clb)
        {
            string wynik = "";
            int dlugosc = clb.CheckedItems.Count;
            for (int i = 0; i < dlugosc - 1; ++i)
            {
                wynik += clb.CheckedItems[i].ToString() + " ";
            }
            wynik += clb.CheckedItems[dlugosc - 1].ToString();
            return wynik;
        }

        private void buttonWyborSciezkiSMDV_Click(object sender, EventArgs e)
        {
            openFileDialog1.ShowDialog();
            textBoxSciezkaDoProgramu.Text = openFileDialog1.FileName;
        }

        private void buttonSciezkaDoMacierzy_Click(object sender, EventArgs e)
        {
            folderBrowserDialog1.ShowDialog();
            textBoxSciezkaDoMacierzy.Text = folderBrowserDialog1.SelectedPath;
        }

        private void buttonSciezkaDoLogow_Click(object sender, EventArgs e)
        {
            folderBrowserDialog1.ShowDialog();
            textBoxSciezkaDoLogow.Text = folderBrowserDialog1.SelectedPath;
        }

        private void buttonSciezkaPySMDV_Click(object sender, EventArgs e)
        {
            openFileDialog1.ShowDialog();
            textBoxSciezkaDoPythonSMDV.Text = openFileDialog1.FileName;
        }

        private void buttonPython_Click(object sender, EventArgs e)
        {
            openFileDialog1.ShowDialog();
            textBoxPython.Text = openFileDialog1.FileName;
        }
    }
}
