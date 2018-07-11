/********************************************************************************
 Written by: LM_1224
 Theme : Launch a Module 
 Date : 30/01/2017

 Serial communication baud rate: 9600bps

 Connection Details:  			
					L-1---->PA0;			L-2---->PA1;
   					R-1---->PA2;			R-2---->PA3;
				PL3 (OC5A) ----> PWM left; 	PL4 (OC5B) ----> PWM right; 

 ADC Connection:
 			  ACD CH.	PORT		Sensor
			  11		PK3		Sharp IR range sensor 3

 PE4 (INT4): External interrupt for left motor position encoder 
 PE5 (INT5): External interrupt for the right position encoder

 Note: 
 
 1. Make sure that in the configuration options following settings are 
 	done for proper operation of the code

 	Microcontroller: atmega2560
 	Frequency: 14745600
 	Optimization: -O0 (For more information read section: Selecting proper optimization 
 						options below figure 2.22 in the Software Manual)


 2. Auxiliary power can supply current up to 1 Ampere while Battery can supply current up to 
 	2 Ampere. When both motors of the robot changes direction suddenly without stopping, 
	it produces large current surge. When robot is powered by Auxiliary power which can supply
	only 1 Ampere of current, sudden direction change in both the motors will cause current 
	surge which can reset the microcontroller because of sudden fall in voltage. 
	It is a good practice to stop the motors for at least 0.5seconds before changing 
	the direction. This will also increase the useable time of the fully charged battery.
	the life of the motor.
 
 3. 5V supply to these motors is provided by separate low drop voltage regulator "5V Servo" which can
 	supply maximum of 800mA current. It is a good practice to move one servo at a time to reduce power surge 
	in the robot's supply lines. Also preferably take ADC readings while servo motor is not moving or stopped
	moving after giving desired position.

 4. It is observed that external interrupts does not work with the optimization level -Os

 5. Distance calculation is for Sharp GP2D12 (10cm-80cm) IR Range sensor

*********************************************************************************/

/********************************************************************************/

#define F_CPU 14745600
#include<avr/io.h>
#include<avr/interrupt.h>
#include<util/delay.h>
#include <math.h> 		//included to support power function


unsigned char data; 					//to store received data from UDR1
unsigned char on_board=0;				//to store no of obj on the bot
unsigned char cmd,txd;					//to store command number
volatile unsigned long int ShaftCountLeft = 0; 		//to keep track of left position encoder
volatile unsigned long int ShaftCountRight = 0; 	//to keep track of right position encoder
volatile unsigned int Degrees; 				//to accept angle in degrees for turning

/********************************************************************************/
/********************************************************************************/

//Function to configure INT4 (PORTE 4) pin as input for the left position encoder
void left_encoder_pin_config (void)
{
	DDRE  = DDRE & 0xEF;  //Set the direction of the PORTE 4 pin as input
	PORTE = PORTE | 0x10; //Enable internal pull-up for PORTE 4 pin
}

//Function to configure INT5 (PORTE 5) pin as input for the right position encoder
void right_encoder_pin_config (void)
{
	DDRE  = DDRE & 0xDF;  //Set the direction of the PORTE 4 pin as input
	PORTE = PORTE | 0x20; //Enable internal pull-up for PORTE 4 pin
}

//Configure PORTB 5 pin for servo motor 1 operation
void servo1_pin_config (void)
{
 DDRB  = DDRB | 0x20;  //making PORTB 5 pin output
 PORTB = PORTB | 0x20; //setting PORTB 5 pin to logic 1
}

//Configure PORTB 6 pin for servo motor 2 operation
void servo2_pin_config (void)
{
 DDRB  = DDRB | 0x40;  //making PORTB 6 pin output
 PORTB = PORTB | 0x40; //setting PORTB 6 pin to logic 1
}

void buzzer_pin_config (void)
{
 DDRC = DDRC | 0x08;		//Setting PORTC 3 as outpt
 PORTC = PORTC & 0xF7;		//Setting PORTC 3 logic low to turnoff buzzer
}

void motion_pin_config (void)
{
 DDRA = DDRA | 0x0F;
 PORTA = PORTA & 0xF0;
 DDRL = DDRL | 0x18;   //Setting PL3 and PL4 pins as output for PWM generation
 PORTL = PORTL | 0x18; //PL3 and PL4 pins are for velocity control using PWM.
}

//ADC pin configuration
void adc_pin_config (void)
{
 DDRF = 0x00; 	//set PORTF direction as input
 PORTF = 0x00; 	//set PORTF pins floating
 DDRK = 0x00;	//set PORTK direction as input
 PORTK = 0x00; 	//set PORTK pins floating
}

/********************************************************************************/
/********************************************************************************/

//Function to initialize ports
void port_init()
{
	motion_pin_config();
	buzzer_pin_config();
	adc_pin_config();
	left_encoder_pin_config(); 	//left encoder pin config
	right_encoder_pin_config(); 	//right encoder pin config
	servo1_pin_config(); 		//Configure PORTB 5 pin for servo motor 1 operation
	servo2_pin_config(); 		//Configure PORTB 6 pin for servo motor 2 operation 
}

//Function To Initialize UART0
// desired baud rate:9600
// actual baud rate:9600 (error 0.0%)
// char size: 8 bit
// parity: Disabled
void uart0_init(void)
{
 UCSR0B = 0x00; //disable while setting baud rate
 UCSR0A = 0x00;
 UCSR0C = 0x06;
 UBRR0L = 0x5F; //set baud rate lo
 UBRR0H = 0x00; //set baud rate hi
 UCSR0B = 0x98;
}

//TIMER1 initialization in 10 bit fast PWM mode  
//prescale:256
// WGM: 7) PWM 10bit fast, TOP=0x03FF
// actual value: 52.25Hz 
void timer1_init(void)
{
	 TCCR1B = 0x00; //stop
	 TCNT1H = 0xFC; //Counter high value to which OCR1xH value is to be compared with
	 TCNT1L = 0x01;	//Counter low value to which OCR1xH value is to be compared with
	 OCR1AH = 0x03;	//Output compare Register high value for servo 1
	 OCR1AL = 0xFF;	//Output Compare Register low Value For servo 1
	 OCR1BH = 0x03;	//Output compare Register high value for servo 2
	 OCR1BL = 0xFF;	//Output Compare Register low Value For servo 2
	 ICR1H  = 0x03;	
	 ICR1L  = 0xFF;
	 TCCR1A = 0xAB; /*{COM1A1=1, COM1A0=0; COM1B1=1, COM1B0=0; COM1C1=1 COM1C0=0}
			For Overriding normal port functionality to OCRnA outputs.
			  {WGM11=1, WGM10=1} Along With WGM12 in TCCR1B for Selecting FAST PWM Mode*/
	 TCCR1C = 0x00;
	 TCCR1B = 0x0C; //WGM12=1; CS12=1, CS11=0, CS10=0 (Prescaler=256)
}

// Timer 5 initialized in PWM mode for velocity control
// Prescale:256
// PWM 8bit fast, TOP=0x00FF
// Timer Frequency:225.000Hz
void timer5_init()
{
	TCCR5B = 0x00;	//Stop
	TCNT5H = 0xFF;	//Counter higher 8-bit value to which OCR5xH value is compared with
	TCNT5L = 0x01;	//Counter lower 8-bit value to which OCR5xH value is compared with
	OCR5AH = 0x00;	//Output compare register high value for Left Motor
	OCR5AL = 0xFF;	//Output compare register low value for Left Motor
	OCR5BH = 0x00;	//Output compare register high value for Right Motor
	OCR5BL = 0xFF;	//Output compare register low value for Right Motor
	OCR5CH = 0x00;	//Output compare register high value for Motor C1
	OCR5CL = 0xFF;	//Output compare register low value for Motor C1
	TCCR5A = 0xA9;	/*{COM5A1=1, COM5A0=0; COM5B1=1, COM5B0=0; COM5C1=1 COM5C0=0}
			  For Overriding normal port functionality to OCRnA outputs.
		  	  {WGM51=0, WGM50=1} Along With WGM52 in TCCR5B for Selecting FAST PWM 8-bit Mode*/
	
	TCCR5B = 0x0B;	//WGM12=1; CS12=0, CS11=1, CS10=1 (Prescaler=64)
}

void left_position_encoder_interrupt_init (void) 	//Interrupt 4 enable
{
	cli(); 						//Clears the global interrupt
	EICRB = EICRB | 0x02; 				// INT4 is set to trigger with falling edge
	EIMSK = EIMSK | 0x10; 				// Enable Interrupt INT4 for left position encoder
	sei();   					// Enables the global interrupt
}

void right_position_encoder_interrupt_init (void) 	//Interrupt 5 enable
{
	cli(); 						//Clears the global interrupt
	EICRB = EICRB | 0x08; 				// INT5 is set to trigger with falling edge
	EIMSK = EIMSK | 0x20; 				// Enable Interrupt INT5 for right position encoder
	sei();  				 	// Enables the global interrupt
}

//Function to Initialize ADC
void adc_init()
{
	ADCSRA = 0x00;
	ADCSRB = 0x00;		//MUX5 = 0
	ADMUX = 0x20;		//Vref=5V external --- ADLAR=1 --- MUX4:0 = 0000
	ACSR = 0x80;
	ADCSRA = 0x86;		//ADEN=1 --- ADIE=1 --- ADPS2:0 = 1 1 0
}

/********************************************************************************/
/********************************************************************************/

void buzzer_on (void)
{
 PORTC = PINC | 0x08;
}

void buzzer_off (void)
{
 PORTC = PINC & 0xF7;
}

/********************************************************************************/

//Function to rotate Servo 1 by a specified angle in the multiples of 1.86 degrees
void servo_1(unsigned char degrees)  
{
 float PositionPanServo = 0;
  PositionPanServo = ((float)degrees / 1.86) + 35.0;
 OCR1AH = 0x00;
 OCR1AL = (unsigned char) PositionPanServo;
}


//Function to rotate Servo 2 by a specified angle in the multiples of 1.86 degrees
void servo_2(unsigned char degrees)
{
 float PositionTiltServo = 0;
 PositionTiltServo = ((float)degrees / 1.86) + 35.0;
 OCR1BH = 0x00;
 OCR1BL = (unsigned char) PositionTiltServo;
}


//servo_free functions unlocks the servo motors from the any angle 
//and make them free by giving 100% duty cycle at the PWM. This function can be used to 
//reduce the power consumption of the motor if it is holding load against the gravity.

void servo_1_free (void) //makes servo 1 free rotating
{
 OCR1AH = 0x03; 
 OCR1AL = 0xFF; //Servo 1 off
}

void servo_2_free (void) //makes servo 2 free rotating
{
 OCR1BH = 0x03;
 OCR1BL = 0xFF; //Servo 2 off
}

/********************************************************************************/

// Function for robot velocity control
void velocity (unsigned char left_motor, unsigned char right_motor)
{
	OCR5AL = (unsigned char)left_motor;
	OCR5BL = (unsigned char)right_motor;
}

//Function used for setting motor's direction
void motion_set (unsigned char Direction)
{
 unsigned char PortARestore = 0;

 Direction &= 0x0F; 			// removing upper nibbel as it is not needed
 PortARestore = PORTA & 0xF0; 		// setting lower direction nibbel to 0
 PORTA = PortARestore | Direction; 	// setting the command to the port
}

void forward (void) //both wheels forward
{
  motion_set(0x06);
}

void back (void) //both wheels backward
{
  motion_set(0x09);
}

void left (void) //Left wheel backward, Right wheel forward
{
  motion_set(0x05);
}

void right (void) //Left wheel forward, Right wheel backward
{
  motion_set(0x0A);
}

void soft_left (void) //Left wheel stationary, Right wheel forward
{
 motion_set(0x04);
}

void soft_right (void) //Left wheel forward, Right wheel is stationary
{
 motion_set(0x02);
}

void soft_left_2 (void) //Left wheel backward, right wheel stationary
{
 motion_set(0x01);
}

void soft_right_2 (void) //Left wheel stationary, Right wheel backward
{
 motion_set(0x08);
}

void stop (void)
{
  motion_set(0x00);
}

/********************************************************************************/

//ISR for right position encoder
ISR(INT5_vect)
{
	ShaftCountRight++;  //increment right shaft position count
}

//ISR for left position encoder
ISR(INT4_vect)
{
	ShaftCountLeft++;  //increment left shaft position count
}

//Function used for turning robot by specified degrees
void angle_rotate(unsigned int Degrees)
{
	float ReqdShaftCount = 0;
	unsigned long int ReqdShaftCountInt = 0;

	ReqdShaftCount = (float) Degrees/ 4.090; 		// division by resolution to get shaft count
	ReqdShaftCountInt = (unsigned int) ReqdShaftCount;
	ShaftCountRight = 0;
	ShaftCountLeft = 0;

	while (1)
	{
		if((ShaftCountRight >= ReqdShaftCountInt) | (ShaftCountLeft >= ReqdShaftCountInt))
		break;
	}
	stop(); //Stop robot
}


//Function used for moving robot forward by specified distance
void linear_distance_mm(unsigned int DistanceInMM)
{
	float ReqdShaftCount = 0;
	unsigned long int ReqdShaftCountInt = 0;

	ReqdShaftCount = DistanceInMM / 5.338; 			// division by resolution to get shaft count
	ReqdShaftCountInt = (unsigned long int) ReqdShaftCount;
	
	ShaftCountRight = 0;
	while(1)
	{
		if(ShaftCountRight > ReqdShaftCountInt)
		{
			break;
		}
	}
	stop(); //Stop robot
}

void forward_mm(unsigned int DistanceInMM)
{
	forward();
	linear_distance_mm(DistanceInMM);
}

void back_mm(unsigned int DistanceInMM)
{
	back();
	linear_distance_mm(DistanceInMM);
}

void left_degrees(unsigned int Degrees)
{
	// 88 pulses for 360 degrees rotation 4.090 degrees per count
	left(); //Turn left
	angle_rotate(Degrees);
}

void right_degrees(unsigned int Degrees)
{
	// 88 pulses for 360 degrees rotation 4.090 degrees per count
	right(); //Turn right
	angle_rotate(Degrees);
}


void soft_left_degrees(unsigned int Degrees)
{
	// 176 pulses for 360 degrees rotation 2.045 degrees per count
	soft_left(); //Turn soft left
	Degrees=Degrees*2;
	angle_rotate(Degrees);
}

void soft_right_degrees(unsigned int Degrees)
{
	// 176 pulses for 360 degrees rotation 2.045 degrees per count
	soft_right();  //Turn soft right
	Degrees=Degrees*2;
	angle_rotate(Degrees);
}

void soft_left_2_degrees(unsigned int Degrees)
{
	// 176 pulses for 360 degrees rotation 2.045 degrees per count
	soft_left_2(); //Turn reverse soft left
	Degrees=Degrees*2;
	angle_rotate(Degrees);
}

void soft_right_2_degrees(unsigned int Degrees)
{
	// 176 pulses for 360 degrees rotation 2.045 degrees per count
	soft_right_2();  //Turn reverse soft right
	Degrees=Degrees*2;
	angle_rotate(Degrees);
}

/********************************************************************************/

//This Function returns the Analog Value of front sharp sensor
unsigned char ADC_Conversion(void)
{
	unsigned char Ch = 0x0B;		// 11 is channel no of front sharp sensor
	unsigned char a;
	ADCSRB = 0x08;
	Ch = Ch & 0x07;  			
	ADMUX= 0x20| Ch;	   		
	ADCSRA = ADCSRA | 0x40;		//Set start conversion bit
	while((ADCSRA & 0x10)==0);	//Wait for ADC conversion to complete
	a=ADCH;
	ADCSRA = ADCSRA|0x10; 		//clear ADIF (ADC Interrupt Flag) by writing 1 to it
	ADCSRB = 0x00;
	return a;
}

// This Function calculates the actual distance in millimeters(mm) from the input
// analog value of Sharp Sensor. 
unsigned int Sharp_GP2D12_estimation(unsigned char adc_reading)
{
	float distance;
	unsigned int distanceInt;
	distance = (int)(10.00*(2799.6*(1.00/(pow(adc_reading,1.1546)))));
	distanceInt = (int)distance;
	if(distanceInt>800)
	{
		distanceInt=800;
	}
	return distanceInt;
}

/********************************************************************************/
/********************************************************************************/

//Finction to set  initial position of arm to verticle open
void init_arm(void)
{
	servo_1(110);			//make arm verticle
	_delay_ms(2000);		//delay
	servo_2(130);			//open grip
	_delay_ms(1000);		//delay
	servo_2_free();
	servo_1_free();
}

//function to pick object
void pick(void)
{
	float sharp,value;
	//adjust position according to the distance of obj from the bot
	sharp = ADC_Conversion();					//Stores the Analog value of front sharp connected to ADC channel 11 into variable "sharp"
	value = Sharp_GP2D12_estimation(sharp);				//Stores Distance calculated in a variable "value".
	
	if(value > 100)			//if bot is greater than 10cm away from obj.
	{
		forward_mm(value-100); 	//move forward 
		_delay_ms(1000);
	}
	else if(value < 80)		//if bot is less than 8cm away from obj.
	{
		back_mm(80-value); 	//move backward 
		_delay_ms(1000);
	}

	if(on_board==0)			//if picking obj is first one
	{
		servo_1(0);			//take arm down
		_delay_ms(2000);		//delay
		servo_2(40);			//close grip
		_delay_ms(1000);		//delay
		servo_2_free();
		servo_1(110);		//lift to verticle
		_delay_ms(1000);	//delay
		servo_1_free();
		on_board += 1;		//No. of obj on bot bocames 1
	}
	else				//if picking obj is second one
	{
		servo_1(180);		//place grabbed obj on the carriage
		_delay_ms(3000);	//delay
		servo_2(130);		//release obj
		_delay_ms(1000);	//delay		
		servo_1(0);		//take arm down
		_delay_ms(2000);	//delay
		servo_2(40);		//close grip
		servo_2_free();
		servo_1(110);		//lift second obj to verticle
		_delay_ms(1000);
		servo_1_free();
		on_board += 1;		//No. of obj on bot becomes 2
	}
}

//function to place obect
void place(void)
{
	//bot has reached to desired position
	if(on_board==1)
	{
		servo_1(5);			//take arm down(don't take to zero)
		_delay_ms(2000);		//delay
		servo_2(130);			//open grip
		_delay_ms(1000);		//delay
		servo_2_free();
		servo_1(110);			//make arm verticle
		_delay_ms(2000);		//delay
		servo_1_free();
		on_board -= 1;			//No. of obj on bot 0
	}
	else
	{
		servo_1(5);			//take arm down(don't take to zero)
		_delay_ms(2000);		//delay
		servo_2(130);			//open grip
		_delay_ms(1000);		//delay
		servo_1(180);			//move arm to back
		_delay_ms(2000);		//delay
		servo_2(40);			//grab onboard obj
		_delay_ms(1000);		//delay
		servo_2_free();
		servo_1(110);			//lift to verticle
		servo_1_free();
		on_board -=1;			//No. of obj on bot 1
	}
}

//function to on/off buzzer
void end_task(void)
{
	buzzer_on();
	_delay_ms(5000);	//wait for 5 seconds
	buzzer_off();
}

/********************************************************************************/
/********************************************************************************/

SIGNAL(SIG_USART0_RECV) 		// ISR for receive complete interrupt
{
	data = UDR0; 			//making copy of data from UDR0 in 'data' variable
	data = data-'0';
	cmd = 1;
}

SIGNAL(SIG_USART0_DATA) 		// ISR for receive complete interrupt
{
	UDR0 = txd; 			//making copy of data from UDR0 in 'data' variable

}

/********************************************************************************/
/********************************************************************************/
 
//Function To Initialize all The Devices
void init_devices(void)
{
 cli(); 		//Clears the global interrupts
 port_init();  		//Initializes all the ports
 timer1_init();
 timer5_init();
 adc_init();
 left_position_encoder_interrupt_init();
 right_position_encoder_interrupt_init();
 uart0_init(); 		//Initailize UART0 for serial communiaction
 sei();   		//Enables the global interrupts
}

/********************************************************************************/
/********************************************************************************/

//Main Function
int main(void)
{
	int l0,m1,n2,o3,c;
	init_devices();
	init_arm();	//initially arm  will be in open state
	l0 = m1 = n2 = o3 = -1;
	c=0;
	while(1)
	{
		if(l0 == -1 && cmd == 1)
		{
			l0 = data;
			cmd == 0;
			txd = data;
		}
		else if(l0 != -1 && m1 == -1 && cmd == 1)
		{
			m1 = data;
			cmd == 0;
			txd = data;
		}	
		else if(m1 != -1 && n2 == -1 && cmd == 1)
		{
			n2 = data;
			cmd == 0;
			txd = data;
			
		}
		else if(n2 != -1 && o3 == -1 && cmd == 1)
		{
			o3 = data;
			txd = data;
			c = 1;
			data = (m1*100)+(n2*10)+o3;
			cmd == 0;
		}			
		if(c==1)
		{
			switch(l0)
			{
				case 1:
					pick();		//pick the obj from the cell
					break;
				case 2:
					place();	//place obj in cell
					break;
				case 3:
					forward_mm(data);
					break;
				case 4:
					left_degrees(data);
					break;
				case 5:
					right_degrees(data);
					break;
				case 6:
					end_task();
					break;
			}
			txd = 'c'; 			//echo data back to PC (as a response saying command executed)
			c = 0;
			l0 = m1 = n2  = o3 = -1;
		}
	}
}

/***********************************************************************************
/***********************************************************************************
