#include <cusp/multiply.h>
#include <cusp/io/matrix_market.h>

#include <cusp/array1d.h>
#include <cusp/array2d.h>

#include <cusp/csr_matrix.h>
#include <cusp/coo_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/hyb_matrix.h>

#include <cusp/exception.h>

#include <Windows.h>
#include <string.h>
#include <iostream>
#include <fstream>

//version 1.2
//author Slawomir Figiel

std::string st(int i) // convert int to string
{
    std::stringstream s;
    s << i;
    return s.str();
}

std::string dateToString(SYSTEMTIME date)
{
	std::string s = "";
	s += st(date.wYear) + "-" + st(date.wMonth) + "-" + st(date.wDay) + " " + st(date.wHour) + ":" + st(date.wMinute) + ":" + st(date.wSecond);
	return s;
}

void przerwij(std::string msg)	// Przerywa i wyswietla wiadomosc. Czeka na Enter.
{
	std::cout << "ERROR: " << msg << std::endl;
	getchar();
	exit(1);
}

template <typename LinearOperator,
          typename MatrixOrVector1,
          typename MatrixOrVector2>
int mnozenie(LinearOperator&  A,
              MatrixOrVector1& B,
              MatrixOrVector2& C,
			  int powtorzenia)				// Wykouje mnozenie podana ilosc razy. Zwraca sredni czas mnozenia.
{
	int start = 0;
	int end = 0;

	start = GetTickCount();
	for (int i = 0; i < powtorzenia; ++i)
	{
		cusp::multiply(A, B, C);
	}
	end = GetTickCount();

	return end - start;
}

 

int main(int argc, char *argv[])
{
	unsigned long startG = GetTickCount64();
	int startInic = GetTickCount();
	SYSTEMTIME dateStart;
	GetLocalTime(&dateStart);
	
	////########################## FOR USERS ###########################
	bool czyCUDA = true;
	bool czyCPU = true;
	bool czyWypisacWynik = true;
	bool czyUtworzycPlik = true;
	bool czyWszystkieInfo = true;															//Warunek niesprawdzany. Zostawiony dla zgodnosci z GUI.

	//std::string sciezka = "E:\\Slawek\\SMVD\\SMDV\\Macierze\\";
	//std::string sciezka = "E:\\Macierze_temp\\";
	//std::string sciezka = "E:\\Moje projekty\\SMDV\\Macierze\\Niedzialajace\\";
	std::string sciezka = "E:\\Moje projekty\\SMDV\\Macierze\\";
	//std::string sciezkaDlaLogow = "E:\\Macierze_temp\\";
	std::string sciezkaDlaLogow = "E:\\Moje projekty\\SMDV\\Logi\\";

	std::string sep = ", ";																	//Seprator rozdzielajacy dane w pliku
	std::string brakDanych = "b/d";

	int ilePowtorzen = 100;																	//Uzywac rozsadnie.

	std::string macierze[100] = {															//Ograniczenie liczby macierzy do 100. Wymagania jezyka. Taki super jezyk niby, a na takiej blahostkce robi wielkie problemy
			//"takaMacierzNieIstnieje.mtx",
			//"qh1484.mtx",
			"dw2048.mtx",
			//"cant.mtx",
			//"cop20k_A.mtx",
			//"mac_econ_fwd500.mtx",
			//"pwtk.mtx",
			//"wbp256.mtx",

			"#@!Koniec!@#"};																//Nie dotykac!

	////########################## END FOR USERS ###########################


	////Jezeli podane zostaly argumenty wiersza polecen:							//TODO: konwersja z char do bool
	if (argc > 11)
	{
		std::string tmp =	argv[1];
		czyCUDA	= (tmp == "True");
		tmp =				argv[2];
		czyCPU = (tmp == "True");;
		tmp =				argv[3];
		czyWypisacWynik	= (tmp == "True");
		tmp =				argv[4];
		czyUtworzycPlik	= (tmp == "True");
		tmp =				argv[5];
		czyWszystkieInfo = (tmp == "True");
		sciezka =			argv[6];
		sciezkaDlaLogow =	argv[7];
		sep =				argv[8];
		brakDanych =		argv[9];
		ilePowtorzen =		std::stoi(argv[10]);

		for (int i = 0; i < argc-11; ++i)
		{
			macierze[i] = argv[11+i];
		}
		macierze[argc-11] = "#@!Koniec!@#";
	}
	else if (argc > 1)
	{
		przerwij("Za malo argumentow.");
	}


	////Sprawdzanie warunkow, zeby niepotrzebnie nie wykonywac
	if ( !( czyCUDA || czyCPU ) || !( czyWypisacWynik || czyUtworzycPlik ) )		////Sprawdzmy czy jestes sprytny
		przerwij("Bledne argumenty. Brak zwracanych rezultatow");

	////Inicjalizacja zmiennych 

	//Czas:
	unsigned int start = 0;											////Pomocnicze zmienne czasu
	unsigned int end = 0;
	unsigned int startObliczenia = 0;
	unsigned int endObliczenia = 0;

	long timeCOO = 0;										////Czasy mnozen GPU - CUDA
	long timeCSR = 0;
	long timeELL = 0;
	long timeHYB = 0;
	long timeCOO_cpu = 0;									////Czasy mnozen CPU
	long timeCSR_cpu = 0;
	long timeELL_cpu = 0;
	long timeHYB_cpu = 0;

	long timeCopyCOO = 0;
	long timeCopyCSR = 0;
	long timeCopyELL = 0;
	long timeCopyHYB = 0;
	long timeCopyCOO_cpu = 0;
	long timeCopyCSR_cpu = 0;
	long timeCopyELL_cpu = 0;
	long timeCopyHYB_cpu = 0;

	long timeReadCSR_cpu = 0;

	long timeInic = 0;										////Czasy wykonania poszczegolnych czesci programu
	long timeCPU = 0;
	long timeCuda = 0;
	

	
	//Szegoly o macierzy
	size_t N = 0;											////wiersze
	size_t kolumny = 0;
	size_t niezerowe = 0;

	//Wektory
	cusp::array1d<float, cusp::device_memory> z;			////Wektor gesty w pamieci GPU
	cusp::array1d<float, cusp::device_memory> y;			////Wektor wynikowy w pamieci GPU
	cusp::array1d<float, cusp::host_memory> x;				////Wektor wynikowy w pamieci CPU (RAM)

	//Inne:
	int indMac = -1;
	std::ofstream plikDanych;	
	std::stringstream errorLog;
	std::string macierz;
	cusp::csr_matrix<int, float, cusp::host_memory> CSR_cpu;

	
	////Tworzenie pliku na dane wyjsciowe
	if (czyUtworzycPlik)
	{
		std::string data;
		SYSTEMTIME ti;
		GetLocalTime(&ti);
		data = st(ti.wYear)+ "_" + st(ti.wMonth) + "_" + st(ti.wDay) + "_" + st(ti.wHour) + "_"+ st(ti.wMinute) + "_" + st(ti.wSecond)+".txt";
		plikDanych.open(sciezkaDlaLogow + data); 
		if(!plikDanych)
		{
			std::cerr<<"Cannot open the output file."<<std::endl;
			przerwij("Nie udalo sie utworzyc pliku na dane wyjsciowe.");
		}
		//NEW: powtorzenia, nazwa, wiersze, kolumny, niezerowe wartosci, czesc inicjalizacyjna (wykonywana tylko raz), obliczenia na CPU,
		//NEW: obliczenia na GPU, czas COO(GPU), czas CSR(GPU), czas ELL(GPU), czas HYB(GPU),
		//						  czas COO(CPU), czas CSR(CPU), czas ELL(CPU), czas HYB(CPU),
		//						  czas kopiowania COO(GPU), czas kopiowania CSR(GPU), czas kopiowania ELL(GPU), czas kopiowania HYB(GPU),
		//						  czas kopiowania COO(CPU), czas odczytu z pliku CSR(GPU), czas kopiowania ELL(CPU), czas kopiowania HYB(CPU),
		plikDanych << "Ile powtorzen" << sep << "Nazwa macierzy" << sep << "Wiersze" << sep << "Kolumny" << sep << "Wartosci niezerowe" << sep << "Inicjalizacja" << sep << "Obliczenia CPU";
		plikDanych << sep << "Obliczenia GPU" << sep << "COO(Gpu)" << sep << "CSR(Gpu)" << sep << "ELL(Gpu)" << sep << "HYB(Gpu)";
		plikDanych << sep << "COO(Cpu)" << sep << "CSR(Cpu)" << sep << "ELL(Cpu)" << sep << "HYB(Cpu)";
		plikDanych << sep << "COO Copy(Gpu)" << sep << "CSR Copy(Gpu)" << sep << "ELL Copy(Gpu)" << sep << "HYB Copy(Gpu)";
		plikDanych << sep << "COO Copy(Cpu)" << sep << "CSR Read from file(Cpu)" << sep << "ELL Copy(Cpu)" << sep << "HYB Copy(Cpu)";
		plikDanych << std::endl;
	}
	
	int endInic = GetTickCount();
	timeInic = endInic - startInic;

	////Wykonywane dla kazdej macierzy oddzielnie
	while (macierze[indMac+1] != "#@!Koniec!@#")
	{
		startObliczenia = GetTickCount();
		indMac++;
		macierz = sciezka + macierze[indMac];						////Ustala fizyczne miejsce macierzy na dysku

		try
		{
				start = GetTickCount();
			cusp::io::read_matrix_market_file(CSR_cpu, macierz);	//Odczyt macierzy z pliku do pamieci RAM. Z tej macierzy konwertowanie na inne. Szybciej niz kazdorazowy odczyt z pliku.	
				end = GetTickCount();

			timeReadCSR_cpu = end - start;
		}
		catch (cusp::format_conversion_exception e)
		{
			errorLog << macierze[indMac] << " : Blad utworzenia macierzy CSR. Przerwano obliczenia dla tego pliku. Opis wyjatku: " << e.what() << std::endl;	//TO DO: Format CSR nieobowiazkowy
			timeReadCSR_cpu = -1;
			continue;
		}
		catch (cusp::io_exception ioe)
		{
			errorLog << macierze[indMac] << " : Nie znaleziono pliku." << std::endl;
			timeReadCSR_cpu = -1;
			continue;
		}
		catch (...)
		{
			errorLog << macierze[indMac] << " : Nieoczekiwany wyjatek podczas odczytu z pliku macierzy CSR. Obliczenia dla tego pliku przerwane." << std::endl;
			timeReadCSR_cpu = -1;
			continue;
		}

		N = CSR_cpu.num_rows;										//Odczyt informacji o macierzy
		kolumny = CSR_cpu.num_cols;
		niezerowe = CSR_cpu.num_entries;
		cusp::array1d<float, cusp::host_memory> w(N);				//Deklaracja wektora gestego w pamieci CPU.
	
		for (float i = 0; i < N; ++i)									//Inicjalizacja wektora gestego.
			w[i] = i+1;

		if (czyCPU)
		{
			try
			{
				cusp::array1d<float, cusp::host_memory> x(N);		//Deklaracja wektora na wynik mnozenia w pamieci CPU.

				try
				{
					cusp::coo_matrix<int, float, cusp::host_memory> COO_cpu;

						start = GetTickCount();
					COO_cpu = CSR_cpu;
						end = GetTickCount();
					timeCopyCOO_cpu = end - start;

					timeCOO_cpu = mnozenie(COO_cpu, w, x, ilePowtorzen);
				}
				catch (cusp::format_conversion_exception e)
				{
					errorLog << macierze[indMac] << " : Blad formatu COO. Opis: " << e.what() << std::endl;
					timeCOO_cpu = -1;
					timeCopyCOO_cpu = -1;
				}

				timeCSR_cpu = mnozenie(CSR_cpu, w, x, ilePowtorzen);		//Nie zapominajcie o mnie

				try
				{
					cusp::ell_matrix<int, float, cusp::host_memory> ELL_cpu;
						start = GetTickCount();
					ELL_cpu = CSR_cpu;
						end = GetTickCount();
					timeCopyELL_cpu = end - start;

					timeELL_cpu = mnozenie(ELL_cpu, w, x, ilePowtorzen);
				}
				catch (cusp::format_conversion_exception e)
				{
					errorLog << macierze[indMac] << " : Blad formatu ELL. Opis: " << e.what() << std::endl;
					timeELL_cpu = -1;
					timeCopyELL_cpu = -1;
				}

				try
				{
					cusp::hyb_matrix<int, float, cusp::host_memory> HYB_cpu;
						start = GetTickCount();
					HYB_cpu = CSR_cpu;
						end = GetTickCount();
					timeCopyHYB_cpu = end - start;

					timeHYB_cpu = mnozenie(HYB_cpu, w, x, ilePowtorzen);
				}
				catch (cusp::format_conversion_exception e)
				{
					errorLog << macierze[indMac] << " : Blad formatu HYB. Opis: " << e.what() << std::endl;
					timeHYB_cpu = -1;
					timeCopyHYB_cpu = -1;
				}
			}
			catch (...)
			{
				errorLog << macierze[indMac] << " : Nieoczekiwany wyjatek w trakcie przetwarzania na CPU. Obliczenia dla macierzy na CPU przerwane. Pewne wyniki moga byc nieprawidlowe." << std::endl;
			}
		}

		endObliczenia = GetTickCount();
		timeCPU = endObliczenia - startObliczenia;

		if (czyCUDA)
		{
			startObliczenia = GetTickCount();
			try
			{
				z = w;													//Kopiowanie wektora gestego do pamieci GPU
				cusp::array1d<float, cusp::device_memory> y(N);			//Inicjalizacja wektora na wynik w pamieci GPU		
			
				try
				{
					cusp::coo_matrix<int, float, cusp::device_memory> COO;
						start = GetTickCount();
					COO = CSR_cpu;
						end = GetTickCount();
					timeCopyCOO = end - start;

					timeCOO = mnozenie(COO, z, y, ilePowtorzen);
				}
				catch (cusp::format_conversion_exception e)
				{
					errorLog << macierze[indMac] << " : Blad formatu COO. Opis: " << e.what() << std::endl;
					timeCOO = -1;
					timeCopyCOO = -1;
				}

				try
				{
					cusp::csr_matrix<int, float, cusp::device_memory> CSR;
						start = GetTickCount();
					CSR = CSR_cpu;
						end = GetTickCount();
					timeCopyCSR = end - start;

					timeCSR = mnozenie(CSR, z, y, ilePowtorzen);
				}
				catch (cusp::format_conversion_exception e)
				{
					errorLog << macierze[indMac] << " : Blad formatu CSR. To sie nie moglo wydarzyc! Opis: " << e.what() << std::endl;
					timeCSR = -1;
					timeCopyCSR = -1;
				}

				try
				{
					cusp::ell_matrix<int, float, cusp::device_memory> ELL;
						start = GetTickCount();
					ELL = CSR_cpu;
						end = GetTickCount();
					timeCopyELL = end - start;

					timeELL = mnozenie(ELL, z, y, ilePowtorzen);
				}
				catch (cusp::format_conversion_exception e)
				{
					errorLog << macierze[indMac] << " : Blad formatu ELL. Opis: " << e.what() << std::endl;
					timeELL = -1;
					timeCopyELL = -1;
				}

				try
				{
					cusp::hyb_matrix<int, float, cusp::device_memory> HYB;
						start = GetTickCount();
					HYB = CSR_cpu;
						end = GetTickCount();
					timeCopyHYB = end - start;

					timeHYB = mnozenie(HYB, z, y, ilePowtorzen);
				}
				catch (cusp::format_conversion_exception e)
				{
					errorLog << macierze[indMac] << " : Blad formatu HYB. Opis: " << e.what() << std::endl;
					timeHYB = -1;
					timeCopyHYB = -1;
				}
			}
			catch (thrust::system::detail::bad_alloc e)
			{
				czyCUDA = false;
				errorLog << "Brak urzadzenia zgodnego z CUDA. Obliczenia na GPU wylaczone. Opis: " << e.what() << std::endl;
				if (! (czyCUDA || czyCPU) )
					przerwij("Brak urzadzenia zgodnego z CUDA. Obliczenia na GPU wylaczone. Brak sposobu wykonania obliczen.");
			}
			catch (...)
			{
				errorLog << macierze[indMac] << " : Nieoczekiwany wyjatek w trakcie przetwarzania na CUDA. Obliczenia dla macierzy na GPU przerwane. Pewne wyniki moga byc nieprawidlowe." << std::endl;
			}
			endObliczenia = GetTickCount();
			timeCuda = endObliczenia - startObliczenia;
		}

		////Wypisanie wynikow:
		if(czyWypisacWynik)
		{
			std::cout << "Macierz: " << macierze[indMac] << " wierszy: " << N << " kolumn: " << kolumny << " niezerowych wartosci: " << niezerowe << std::endl;
			std::cout << "Czas wykonania inicjalizacji: " << timeInic << " ms, obliczen na CPU: " << timeCPU << "ms, na GPU: " << timeCuda << " ms" << std::endl;
			std::cout << "Czas wykonania mnozenia przez CUSP dla: " << ilePowtorzen << " powtorzen." << std::endl;
			if (czyCUDA)
			{
				std::cout << "GPU - CUDA: " << std::endl;
				std::cout << "Format COO: "<< timeCOO << " ms = " << timeCOO / 1000 << " sec = " << timeCOO/60000 << " min" << std::endl;
				std::cout << "Format CSR: "<< timeCSR << " ms = " << timeCSR / 1000 << " sec = " << timeCSR/60000 << " min" << std::endl;
				std::cout << "Format ELL: "<< timeELL << " ms = " << timeELL / 1000 << " sec = " << timeELL/60000 << " min" << std::endl;
				std::cout << "Format HYB: "<< timeHYB << " ms = " << timeHYB / 1000 << " sec = " << timeHYB/60000 << " min" << std::endl;
			}
			if (czyCPU)
			{
				std::cout << "CPU" << std::endl;
				std::cout << "Format COO(CPU): "<< timeCOO_cpu << " ms = " << timeCOO_cpu/1000 << " sec = " << timeCOO_cpu/60000 << " min" << std::endl;
				std::cout << "Format CSR(CPU): "<< timeCSR_cpu << " ms = " << timeCSR_cpu/1000 << " sec = " << timeCSR_cpu/60000 << " min" << std::endl;
				std::cout << "Format ELL(CPU): "<< timeELL_cpu << " ms = " << timeELL_cpu/1000 << " sec = " << timeELL_cpu/60000 << " min" << std::endl;
				std::cout << "Format HYB(CPU): "<< timeHYB_cpu << " ms = " << timeHYB_cpu/1000 << " sec = " << timeHYB_cpu/60000 << " min" << std::endl;
			}
			std::cout << std::endl;
			std::cout << "Czas kopiowania danych z formatu CSR zainicjalizowanego na CPU dla:" << std::endl;
			if (czyCUDA)
			{
				std::cout << "GPU - CUDA: " << std::endl;
				std::cout << "Format COO: "<< timeCopyCOO << " ms = " << timeCopyCOO / 1000 << " sec = " << timeCopyCOO/60000 << " min" << std::endl;
				std::cout << "Format CSR: "<< timeCopyCSR << " ms = " << timeCopyCSR / 1000 << " sec = " << timeCopyCSR/60000 << " min" << std::endl;
				std::cout << "Format ELL: "<< timeCopyELL << " ms = " << timeCopyELL / 1000 << " sec = " << timeCopyELL/60000 << " min" << std::endl;
				std::cout << "Format HYB: "<< timeCopyHYB << " ms = " << timeCopyHYB / 1000 << " sec = " << timeCopyHYB/60000 << " min" << std::endl;
			}
			if (czyCPU)
			{
				std::cout << "CPU (dla macierzy CSR podano czas kopiowania z pliku):" << std::endl;
				std::cout << "Format COO(CPU): "<< timeCopyCOO_cpu << " ms = " << timeCopyCOO_cpu/1000 << " sec = " << timeCopyCOO_cpu/60000 << " min" << std::endl;
				std::cout << "Format CSR(CPU): "<< timeReadCSR_cpu << " ms = " << timeReadCSR_cpu/1000 << " sec = " << timeReadCSR_cpu/60000 << " min" << std::endl;
				std::cout << "Format ELL(CPU): "<< timeCopyELL_cpu << " ms = " << timeCopyELL_cpu/1000 << " sec = " << timeCopyELL_cpu/60000 << " min" << std::endl;
				std::cout << "Format HYB(CPU): "<< timeCopyHYB_cpu << " ms = " << timeCopyHYB_cpu/1000 << " sec = " << timeCopyHYB_cpu/60000 << " min" << std::endl;
			}
		}
	
		if(czyUtworzycPlik)
		{
			//OLD: powtorzenia, nazwa, wiersze, kolumny, niezerowe wartosci, inicjalizacja wektora, inicjalizacja macierzy z pliku
			//NEW: powtorzenia, nazwa, wiersze, kolumny, niezerowe wartosci, czesc inicjalizacyjna (wykonywana tylko raz), obliczenia na CPU
			plikDanych << ilePowtorzen << sep << macierze[indMac] << sep << N << sep << kolumny << sep << niezerowe << sep << timeInic << sep << timeCPU;
			if (czyCUDA)
			{
				//OLD: czas COO(GPU), czas CSR(GPU), czas ELL(GPU), czas HYB(GPU)
				//NEW: obliczenia na GPU, czas COO(GPU), czas CSR(GPU), czas ELL(GPU), czas HYB(GPU)
				plikDanych << sep << timeCuda << sep << timeCOO << sep << timeCSR << sep << timeELL << sep << timeHYB;
			}
			else
			{
				plikDanych << sep << brakDanych << sep << brakDanych << sep << brakDanych << sep << brakDanych << sep << brakDanych; 
			}
			if (czyCPU)
			{
				//czas COO(CPU), czas CSR(CPU), czas ELL(CPU), czas HYB(CPU)
				plikDanych << sep << timeCOO_cpu << sep << timeCSR_cpu << sep << timeELL_cpu << sep << timeHYB_cpu;
			}
			else
			{
				plikDanych << sep << brakDanych << sep << brakDanych << sep << brakDanych << sep << brakDanych; 
			}
			if (czyCUDA)
			{
				//czas kopiowania COO, czas kopiowania CSR, czas kopiowania ELL, czas kopiowania HYB
				plikDanych << sep << timeCopyCOO << sep << timeCopyCSR << sep << timeCopyELL << sep << timeCopyHYB;
			}
			else
			{
				plikDanych << sep << brakDanych << sep << brakDanych << sep << brakDanych << sep << brakDanych; 
			}
			if (czyCPU)
			{
				//czas kopiowania COO(CPU), czas odczytu z pliku CSR(CPU), czas kopiowania ELL(CPU), czas kopiowania HYB(CPU)
				plikDanych << sep << timeCopyCOO_cpu << sep << timeReadCSR_cpu << sep << timeCopyELL_cpu << sep << timeCopyHYB_cpu;
			}
			else
			{
				plikDanych << sep << brakDanych << sep << brakDanych << sep << brakDanych << sep << brakDanych; 
			}
			plikDanych << std::endl;
		}
	}
	SYSTEMTIME dateEnd;
	GetLocalTime(&dateEnd);
	unsigned long endG = GetTickCount();

	if (argc == 1)
	{
		errorLog << "Wykonano dla parametrow wewnetrznych programu." << std::endl;
	}

	if (czyUtworzycPlik)
	{
		if (errorLog.str() != "")
		{
			plikDanych << std::endl << std::endl << "ERROR LOG:"<< std::endl;
			plikDanych << errorLog.str() << std::endl;
			plikDanych.close();
		}
	}

	if (czyWypisacWynik)
	{
		if (errorLog.str() != "")
		{
			std::cout << std::endl << "ERROR LOG:"<< std::endl;
			std::cout << errorLog.str() << std::endl;
		}
		std::cout << std::endl << "Wykonalem. Rozpoczeto: " << dateToString(dateStart) << ", zakonczono: " << dateToString(dateEnd) << std::endl << "Czas wykonania: " << (endG - startG)/60000 << " min " << std::endl;
		getchar();
	}

	return 0;
}
