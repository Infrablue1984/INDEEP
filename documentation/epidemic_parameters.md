Im folgenden werden die einstellbaren Parameter des Agenten basierten Modells erläutert.

Faktor der nicht gemeldeten Infektionen (Dunkelziffer)

Die Dunkelziffer beschreibt das Missverhältnis von diagnostizierten (gemeldeten) Krankheitsfällen zur Anzahl von tatsächlich erkrankten Bürger:innen.

Eine Dunkelziffer von 3 entspricht dabei der Situation von drei mal soviel Erkrankten, wie eigentlich gemeldet sind.

Wie viel Prozent der Bevölkerung sind (ohne Dunkelziffer) infiziert?

Hier wird der prozentuale Anteil der offiziell gemeldeten infizierten Bevölkerung eingegeben.

Wie viel Prozent der Bevölkerung sind immun gegen die Krankheit?

Dieser Parameter ist auch bekannt unter dem Namen der **Seroprävalenz**.

Hier handelt es sich um den prozentualen Anteil der Bevölkerung, der eine natürliche Immunität gegen die Krankheit aufweist.

Wie viel Prozent der Bevölkerung sind geimpft?

Hier wird der prozentuale Anteil der geimpften Bevölkerung angegeben.

Zum jetzigen Zeitpunkt beinhaltet das Modell keine Neu- sowie Auffrischungsimpfungen von Bürger:innen.

Für wie viel Prozent der Bevölkerung gibt es Krankenhausbetten?

Dieser Parameter gibt, mittels prozentualem Anteil an der Gesamtbevölkerung, die Anzahl der Krankenhausbetten in den zu simulierenden Regionen an.

Dieser Wert entspricht damit der Anzahl von Bürger:innen, die zum selben Zeitpunkt im Krankenhaus behandelt werden können.

Parameter $\mu$ und $\sigma$ der Viruslast $c_v$ in der Bevölkerung

Die Viruslast beschreibt die im Speichel einer infizierten Person vorkommende Menge an Viren und wird in diesem Modell als Konzentration der RNA Kopien pro \si{\milli\liter angegeben.

Innerhalb einer Bevölkerung kann die Viruslast von Mensch zu Mensch, aber auch innerhalb eines Krankheitsverlaufs stark variieren.

In diesem Modell wird jedem Agenten daher eine Standard-Viruslast zugeordnet, die anhand einer logarithmischen Normalverteilung bestimmt wird.

Diese Verteilung wird festgelegt durch zwei Parameter $\mu$ und $\sigma$, die im Tool separat eingestellt werden können.

Dabei wird, wie in der Medizin üblich, die Einheit von $\log_{10(\text{RNA Kopien pro \si{\milli\liter)$ gebraucht.

Zu Beginn der Infektiösität eines Agenten wird seine Viruslast für 4 Tage, durch Verdoppelung seiner Standard-Viruslast, erhöht.

Anzahl der RNA Kopien pro infektiösem Quantum

Ein infektiöses Quantum wird in der Wissenschaft benutzt, um die Menge der Viren, die für eine Ansteckung notwendig sind, zu quantifizieren.

Dabei wird die Wahrscheinlichkeit für die Ansteckung einer suszeptiblen Person durch die Menge der von ihr inhalierten infektiösen Quanten bestimmt.

Der hier eingegebene Parameter Wert entspricht dem Umrechnungsfaktor (\textit{*engl. conversion factor*) von RNA Kopien des Virus zu infektiösem Quantum.

Weiterführende Informationen zur Modellierung mit infektiösen Quanten finden sich in \cite{BUONANNO2020106112, \cite{Mikszewski2021.01.26.21250580 und \cite{To2010.

Mittelwert und Standardabweichung der \newline Tröpfchen-Volumenkonzentration beim Sprechen

Die Tröpfchen-Volumenkonzentration einer Person gibt an, zu welchem Teil ihre ausgeatmete Luft aus Tröpfchen bzw. Aerosolen besteht.

Dabei wird als Einheit \si{\milli\liter pro \si{\cubic\metre benutzt.

Um eine Varianz in der Bevölkerung zu ermöglichen, bekommt jeder Agent eine Standard-Tröpfchen-Volumenkonzentration zugewiesen, die anhand einer Normalverteilung ermittelt wird.

Der Mittelwert und die Standardabweichung, die diese Normalverteilung beschreiben, werden als Parameter angegeben.

Infektionsparameter $\gamma$ und $\beta$ für direkte Kontakte

Um die Ansteckung über direkte Kontakte einheitlich mit der Ansteckung über Aerosole zu gestalten wurden zwei abstrakte Parameter $\gamma$ und $\beta$ eingeführt.

Dem Ansatz von \cite{BUONANNO2020106112 zur Bestimmung der inhalierten infektiösen Quanta folgend, enthält das Modell die Berechnung mittels empfangener Quantendosis (\textit{*engl. dose of received quanta*).

Diese wird mit der folgenden Formel berechnet:

\begin{align\label{quantendosis_cp

​    D_q = f_s \cdot f_m \cdot IR_e \cdot IR_r \cdot QEL \cdot \gamma \int_{0^{T\left(1-e^{-\beta\cdot t\right)dt,

\end{align

wobei $f_s$ und $f_m$ als Faktoren das Abstandhalten und Tragen einer Maske beinhalten, $IR_e$ und $IR_r$ die Inhalationsrate von Emitter und Empfänger (\textit{*engl. receiver*) beschreiben, $QEL$ die Quanten-Emissionslast und $T$ die Dauer des Kontaktes angibt.

Da die Parameter $\gamma$ und $\beta$ keine physikalische Bedeutung haben, müssen diese für eine belastbare Simulation kalibriert werden.

Ablagerungsrate von Aerosolen auf Oberflächen

Die Ablagerungsrate von Aerosolen ist gegeben durch

\begin{align*

​    k = \frac{1{t_d,

\end{align*

wobei die Ablagerungszeit $t_d$ eines Aerosols als Quotient von Ablagerungsgeschwindigkeit und Höhe der Emissionsquelle bestimmt wird.

Als Einheit wird im Modell \si{\per\hour genutzt.

Virusinaktivierungsrate

Die Inaktivierung von Viren wird beschrieben durch den exponentiellen Abfall

\begin{align*

​    N(t) = N_0e^{-kt,

\end{align*

wobei $N_0$ die Anzahl der Viren zu Beginn darstellt.

Die Virusinaktivierungsrate (oder allgemein Eliminationskonstante) $k$ kann dabei durch die Halbwertszeit $t_{1/2$ folgendermaßen ausgedrückt werden

\begin{align*

​    k = \frac{\ln 2{t_{1/2.

\end{align*

Die Virusinaktivierungsrate wird dabei in der Einheit \si{\per\hour angegeben.

Anteil der Viren, die beim Tragen einer Maske vom Emitter, trotzdem als Aerosole in den Raum ausgestoßen werden

Für infektiöse Agenten, die beim Kontakt mit anderen Agenten eine Maske tragen (siehe \ref{ssec:maßnahmen), gibt dieser Wert den Anteil der Aerosole an, die nicht von der Maske aufgefangen werden und dadurch trotzdem in den Raum gelangen.

Anteil der Viren, die beim Tragen einer Maske vom Empfänger, trotzdem eingeatmet werden

Für suszeptible Agenten, die beim Kontakt mit anderen Agenten eine Maske tragen (siehe \ref{ssec:maßnahmen), gibt dieser Wert den Anteil der Aerosole an, die nicht von der Maske aufgefangen werden und dadurch dennoch vom Agenten inhaliert werden.

Reduktionsfaktor für die Wahrscheinlichkeit der Infektionsübertragung, wenn beim Kontakt Abstand gehalten wird

Für jeden direkten Kontakt von einem ansteckenden und einem suszeptiblen Agenten wird eine Wahrscheinlichkeit für die Übertragung der Infektion ausgerechnet.

Wenn beide Agenten auf sozialen Abstand achten (siehe \ref{ssec:maßnahmen) wird die Übertragungswahrscheinlichkeit mit dem angegebenen Reduktionsfaktor multipliziert.

Wenn nur einer der beiden Agenten Abstand hält, wird der Mittelwert aus dem angegeben Wert und 1 multipliziert.

Reduktionsfaktor für die Viruslast bei geimpften Personen

Bei geimpften Agenten wird die Standard-Viruslast mit dem angegebenen Wert multipliziert und somit reduziert.

Reduktionsfaktor für die Wahrscheinlichkeit eines schweren bzw. kritischen Verlaufs bei geimpften Personen

Für geimpfte Agenten wird die Wahrscheinlichkeit für einen schweren bzw. kritischen Verlauf durch Multiplikation mit dem angegebenen Wert reduziert.

Erhöhungsfaktor für die Wahrscheinlichkeit eines kritischen Verlaufs bzw. das Sterben, wenn eine schwer kranke Person keinen Krankenhausplatz bekommt

Wenn die Krankenhäuser überlastet sind, wird bei den Agenten, die aus Platzmangel nicht im Krankenhaus behandelt werden können, die Wahrscheinlichkeit für einen kritischen und tödlichen Verlauf mit dem angegebenen Faktor multipliziert.

Mittelwert und Standardabweichung des Übergangs zwischen zwei Krankheitszuständen in Tagen

Für jeden Übergang zwischen zwei Krankheitszuständen eines Agenten wird eine Übergangsdauer anhand einer Normalverteilung bestimmt.

Diese Normalverteilung ist charakterisiert durch die angegebenen Werte für Mittelwert und Standardabweichung.

Wahrscheinlichkeit für einen symptomatischen, schweren, kritischen und tödlichen Krankheitsverlauf

Jedem Agenten wird abhängig von seinem Alter eine Wahrscheinlichkeit für einen symptomatischen, schweren, kritischen und tödlichen Krankheitsverlauf zugeordnet.
