# ADHD_Study

Attention deficit hyperactivity disorder (ADHD / ADD) is a life-long neurodevelopmental disorder. ADHD may lead to serious consequences including school dropout, depression, problems with relationships, substance abuse, delinquency, and job failure. The CDC reports that the estimated number of children in the United States diagnosed with ADHD is 6.1 million (9.4%). Boys are more likely to be diagnosed with ADHD than girls (12.9% compared to 5.6%). 

Underdiagnosis in female children is a serious problem for those who suffer silently, perhaps lucky enough to find out in adulthood they have a condition that has been holding them back. I was one of those who was undiagnosed in plain sight as a child and impacted as an adult; dropping out of college twice, struggling at work, and failing wildly at relationships. I received a totally different outlook on who I was after receiving a diagnosis in adulthood. According to the research done by Dr. Quinn and Sharon Wigal, Ph.D., associate clinical professor of pediatrics at the University of California at Irvine, four out of 10 teachers report more difficulty in recognizing ADHD symptoms in girls than in boys. (Connolly, 2019) We need to learn how to separate out physical hyperactivity (prevalent in boys) to identify cognitive hyperactivity (prevalent in girls). 

I created a model that could potentially help identify these undiagnosed children using the CDC National Children’s Health Survey (NCHS) 2011 - 2012 dataset, which included 366 variables. I explored several approaches for determining which data to train the models on to identify females who would be more likely to remain undiagnosed if they are suffering from the disorder. 

1. Full set

2. SEX = Female only

3. K2Q34A = 0

I believe approach #3 is best and will explain the benefits of using this subset in more detail.

