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
            try
            {
                #region odczyt ustawień z pliku
                StreamReader czytaj = File.OpenText("UstawieniaSMDV.txt");
                string linijka = czytaj.ReadLine();
                if (linijka == "null")
                    textBoxSciezkaDoProgramu.Text = "";
                else
                    textBoxSciezkaDoProgramu.Text = linijka;
                textBoxSciezkaDoMacierzy.Text = czytaj.ReadLine();
                textBoxSciezkaDoLogow.Text = czytaj.ReadLine();
                textBoxSeparator.Text = czytaj.ReadLine();
                textBoxSymbolBrakDanych.Text = czytaj.ReadLine();
                textBoxPowtorzenia.Text = czytaj.ReadLine();
                string czyCUDA = czytaj.ReadLine();
                string czyCPU = czytaj.ReadLine();
                string czyDodatkoweDane = czytaj.ReadLine();
                string czyKomunikatyNaKonsoli = czytaj.ReadLine();
                string czyUtworzycLog = czytaj.ReadLine();
                
                if (czyCUDA == "true")
                    checkBoxCUDA.Checked = true;
                else
                    checkBoxCUDA.Checked = false;

                if (czyCPU == "true")
                    checkBoxCPU.Checked = true;
                else
                    checkBoxCPU.Checked = false;

                if (czyDodatkoweDane == "true")
                    checkBoxBrakDanych.Checked = true;
                else
                    checkBoxBrakDanych.Checked = false;

                if (czyKomunikatyNaKonsoli == "true")
                    checkBoxKonsola.Checked = true;
                else
                    checkBoxKonsola.Checked = false;

                if (czyUtworzycLog == "true")
                    checkBoxPlik.Checked = true;
                else
                    checkBoxPlik.Checked = false;

                szukajMacierzy();                           //Przeszukuje katalog z macierzami pod kątem plików .mtx

                List<string> macierze = czytaj.ReadToEnd().Split(new string[] { "\r\n" }, StringSplitOptions.None).ToList<string>();

                zaznaczCheckedList(checkedListBoxMacierze, macierze);

                czytaj.Close();
                #endregion
            }
            catch (IOException ioe) 
            {
                //To specjalnie. W przypadku braku pliku nie rób nic.
            }     
        }
        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            #region zapis ustawień do pliku
            StreamWriter pisz = new StreamWriter("UstawieniaSMDV.txt");
            if (textBoxSciezkaDoProgramu.Text == "")
                pisz.WriteLine("null");
            else
                pisz.WriteLine(textBoxSciezkaDoProgramu.Text);
            pisz.WriteLine(textBoxSciezkaDoMacierzy.Text);
            pisz.WriteLine(textBoxSciezkaDoLogow.Text);
            pisz.WriteLine(textBoxSeparator.Text);
            pisz.WriteLine(textBoxSymbolBrakDanych.Text);
            pisz.WriteLine(textBoxPowtorzenia.Text);

            if (checkBoxCUDA.Checked)
                pisz.WriteLine("true");
            else
                pisz.WriteLine("false");
            if (checkBoxCPU.Checked)
                pisz.WriteLine("true");
            else
                pisz.WriteLine("false");
            if (checkBoxBrakDanych.Checked)
                pisz.WriteLine("true");
            else
                pisz.WriteLine("false");
            if (checkBoxKonsola.Checked)
                pisz.WriteLine("true");
            else
                pisz.WriteLine("false");
            if (checkBoxPlik.Checked)
                pisz.WriteLine("true");
            else
                pisz.WriteLine("false");

            for (int i = 0; i < checkedListBoxMacierze.CheckedItems.Count; ++i)
            {
                pisz.WriteLine(checkedListBoxMacierze.CheckedItems[i].ToString());
            }
            pisz.Close();
            #endregion
        }

        private void buttonStart_Click(object sender, EventArgs e)
        {
            #region obsluga błędów
            labelLog.Text = "Brak błędów.";
            try
            {
                int powtorzenia = 0;
                bool czyPowtorzenia = Int32.TryParse(textBoxPowtorzenia.Text, out powtorzenia);
                if (!(czyPowtorzenia && (powtorzenia > 0)))
                {
                    throw new InvalidDataException("Błąd: Powtórzenia.");
                }
                else if (!(checkBoxCPU.Checked || checkBoxCUDA.Checked))
                {
                    throw new InvalidDataException("Błąd: Brak sposobu wykonania obliczeń.");
                }
                else if (!(checkBoxKonsola.Checked || checkBoxPlik.Checked))
                {
                    throw new InvalidDataException("Błąd: Program nic nie zwraca.");
                }
                else if (textBoxSciezkaDoMacierzy.Text.Trim() == "")
                {
                    throw new InvalidDataException("Błąd: Ścieżka macierzy jest pusta.");
                }
                else if (textBoxSciezkaDoLogow.Text.Trim() == "")
                {
                    throw new InvalidDataException("Błąd: Ścieżka logów jest pusta.");
                }
                else if (textBoxSeparator.Text.Trim() == "")
                {
                    throw new InvalidDataException("Błąd: Brak separatora.");
                }
                else if (textBoxSymbolBrakDanych.Text.Trim() == "")
                {
                    throw new InvalidDataException("Błąd: Brak symbolu oznaczającego brak danych.");
                }
                else if (checkedListBoxMacierze.CheckedItems.Count < 1)
                {
                    throw new InvalidDataException("Błąd: Brak macierzy.");
                }
            }
            catch (InvalidDataException ide)
            {
                labelLog.Text = ide.Message;
                return;
            }
            
            #endregion

            string wyjscie = string.Format("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}",
                                            checkBoxCUDA.Checked,           //0
                                            checkBoxCPU.Checked,            //1
                                            checkBoxKonsola.Checked,        //2
                                            checkBoxPlik.Checked,           //3
                                            checkBoxBrakDanych.Checked,     //4
                                            textBoxSciezkaDoMacierzy.Text,  //5
                                            textBoxSciezkaDoLogow.Text,     //6
                                            textBoxSeparator.Text,          //7
                                            textBoxSymbolBrakDanych.Text,   //8
                                            textBoxPowtorzenia.Text,        //9
                                            checkedElementsInCheckedListToString(checkedListBoxMacierze)//10                       //10
                                            );
            try
            {
                Process myProcess = new Process();
                myProcess.StartInfo.FileName = textBoxSciezkaDoProgramu.Text+@"SMDV.exe";
                //Do not receive an event when the process exits.
                myProcess.EnableRaisingEvents = false;
                // Parameters
                myProcess.StartInfo.Arguments = wyjscie;
                // Modify the following to hide / show the window
                myProcess.StartInfo.CreateNoWindow = false;
                myProcess.StartInfo.UseShellExecute = true;
                myProcess.StartInfo.WindowStyle = ProcessWindowStyle.Maximized;
                myProcess.Start();
            }
            catch (Exception fe)
            {
                labelLog.Text = "Błąd: Brak pliku programu SMDV.exe.";
                return;
            }
        }

        

        private void buttonSzukaj_Click(object sender, EventArgs e)
        {
            szukajMacierzy();
        }

        private void szukajMacierzy()                                               //TO DO: Funkcja do poprawienia
        {
            string sciezka = textBoxSciezkaDoMacierzy.Text;
            List<string> macierzePliki;
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
                macierzePliki[i] = macierzePliki[i].Replace(sciezka, "");
            }
            #endregion

            #region usuwanie nieistniejacych elementow
            for (int i = 0; i < checkedListBoxMacierze.Items.Count; ++i)
            {
                if (stringWLiscie(checkedListBoxMacierze.Items[i].ToString(), macierzePliki))
                {
                    macierzePliki.Remove(checkedListBoxMacierze.Items[i].ToString());
                }
                else
                {
                    checkedListBoxMacierze.Items.RemoveAt(i);
                }
            }
            #endregion

            #region dodawanie nowych elementow
            foreach (string m in macierzePliki)
            {
                checkedListBoxMacierze.Items.Add(m);
            }
            #endregion

        }

        private bool stringWLiscie(string s, List<string> lista)
        {
            foreach (string item in lista)
            {
                if (s == item)
                    return true;      
            }
            return false;
        }

        private void zaznaczCheckedList(CheckedListBox clb, List<string> lista)
        {
            for (int i = 0; i < clb.Items.Count; ++i)
            {
                if (stringWLiscie(clb.Items[i].ToString(), lista))
                {
                    clb.SetItemChecked(i, true);    
                }
            }
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
    }
}
